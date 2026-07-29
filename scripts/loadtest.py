"""부하 스크립트. 동시 답변 제출로 429를 재현하고 대응 효과를 잰다.

    uv run python -m scripts.loadtest --requests 60 --concurrency 30 --label "대응 후"

세션 생성과 문제 출제는 준비 단계라 미리 끝내고, **답변 제출만 동시에 쏜다.**
답변 제출이 임베딩과 LLM을 부르는 유일한 경로이기 때문이다. 준비 단계를 섞으면
측정값에 DB 왕복이 섞여 무엇 때문에 느린지 알 수 없게 된다.

서버 쪽 카운터(/metrics)를 앞뒤로 찍어 차이를 낸다. **클라이언트가 보는
성공률만으로는 429가 몇 번 났는지 알 수 없다.** 재시도가 흡수해버려서
밖에서는 "느렸지만 성공"으로만 보인다.
"""

import argparse
import asyncio
import time
from typing import Any

import httpx


async def _create_prepared_session(
    client: httpx.AsyncClient, quiz_set_id: int, persona_id: int
) -> int | None:
    """세션을 만들고 첫 문제까지 받아둔다. 답변할 수 있는 상태로 만드는 것."""
    created = await client.post(
        "/sessions", json={"quiz_set_id": quiz_set_id, "persona_id": persona_id}
    )
    if created.status_code != 201:
        return None
    session_id = int(created.json()["data"]["session_id"])
    served = await client.post(f"/sessions/{session_id}/next")
    if served.status_code != 200:
        return None
    return session_id


async def _submit(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    session_id: int,
    answer: str,
) -> tuple[int, float]:
    async with semaphore:
        started = time.perf_counter()
        try:
            response = await client.post(
                f"/sessions/{session_id}/answer", json={"answer": answer}
            )
            status = response.status_code
        except httpx.HTTPError:
            status = 0  # 연결 자체가 실패
        return status, time.perf_counter() - started


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _diff_counters(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    before_counters = before.get("counters", {})
    after_counters = after.get("counters", {})
    return {
        key: value - before_counters.get(key, 0)
        for key, value in after_counters.items()
        if value - before_counters.get(key, 0) != 0
    }


async def run(args: argparse.Namespace) -> None:
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=timeout) as client:
        metrics_before = (await client.get("/metrics")).json()["data"]

        print(f"준비 중: 세션 {args.requests}개")
        prepared = await asyncio.gather(
            *[
                _create_prepared_session(client, args.quiz_set, args.persona)
                for _ in range(args.requests)
            ]
        )
        sessions = [s for s in prepared if s is not None]
        if len(sessions) < args.requests:
            print(f"경고: 세션 준비 실패 {args.requests - len(sessions)}건")

        print(f"발사: 동시 {args.concurrency}, 총 {len(sessions)}건")
        semaphore = asyncio.Semaphore(args.concurrency)
        started = time.perf_counter()
        results = await asyncio.gather(
            *[_submit(client, semaphore, sid, args.answer) for sid in sessions]
        )
        wall = time.perf_counter() - started

        metrics_after = (await client.get("/metrics")).json()["data"]

    statuses: dict[int, int] = {}
    for status, _ in results:
        statuses[status] = statuses.get(status, 0) + 1
    latencies = [seconds for _, seconds in results]
    succeeded = statuses.get(200, 0)
    total = len(results)

    print()
    print(f"== {args.label} ==")
    print(f"총 {total}건, 소요 {wall:.2f}초, 처리량 {total / wall:.1f}건/초")
    print(f"성공률 {succeeded / total * 100:.1f}% ({succeeded}/{total})")
    print(f"지연 p50 {_percentile(latencies, 0.5):.2f}초 / "
          f"p95 {_percentile(latencies, 0.95):.2f}초 / "
          f"최대 {max(latencies):.2f}초")
    print(f"상태 코드 {dict(sorted(statuses.items()))}")
    print()
    print("서버 카운터 증분 (클라이언트에서는 안 보이는 값)")
    for key, value in sorted(_diff_counters(metrics_before, metrics_after).items()):
        print(f"  {key}: {value}")

    print()
    print("문서 붙여넣기용 한 줄")
    print(
        f"| {args.label} | {total} | {args.concurrency} | "
        f"{succeeded / total * 100:.1f}% | "
        f"{_percentile(latencies, 0.5):.2f} | {_percentile(latencies, 0.95):.2f} | "
        f"{_diff_counters(metrics_before, metrics_after).get('llm.rate_limited', 0)} | "
        f"{_diff_counters(metrics_before, metrics_after).get('llm.giveup', 0)} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="답변 제출 부하 실험")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=40, help="총 답변 수")
    parser.add_argument("--concurrency", type=int, default=20, help="동시 제출 수")
    parser.add_argument("--quiz-set", type=int, default=1)
    parser.add_argument("--persona", type=int, default=1)
    parser.add_argument(
        "--answer",
        default="글쎄요 잘 모르겠는데요",
        help="애매한 답변일수록 LLM 2차 판정으로 넘어가 쿼터를 더 쓴다",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--label", default="측정", help="결과 표에 쓸 이름")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
