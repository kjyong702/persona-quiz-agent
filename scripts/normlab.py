"""정규화 템플릿이 실제로 값을 하는가, 그리고 패러프레이즈 불변성이 지켜지는가.

    uv run python -m scripts.normlab

**두 가지를 잰다.**

1. **정규화 전후 비교** — `norm-v1`을 적용한 것과 원문 그대로를 각각 임베딩해
   유사도가 어떻게 달라지는지 본다. `judge-normalization.md`에 "표기 차이가
   유사도에 섞이면 임계값이 의미를 잃는다"고 적어뒀는데 **재본 적이 없다.**

2. **패러프레이즈 불변성** — 같은 뜻을 다르게 쓴 답변들(`pair_id`로 묶인 것)이
   실제로 비슷한 유사도를 받는지. **뜻이 같으면 값도 비슷해야 한다**는 것이
   이 판정 방식의 전제인데, 그 전제가 얼마나 지켜지는지는 별개 문제다.

임베딩은 캐시한다(`scripts/measure.py`와 같은 파일). 정규화 전 문자열은
새로 불러야 하므로 첫 실행에만 비용이 든다.
"""

import asyncio
import json
import statistics as stat
from pathlib import Path

from app.core import embedding, normalization, vector_store
from app.core.config import settings

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "eval" / "dataset.json"
CACHE = ROOT / "eval" / ".embedding-cache.json"
CHUNK = 100


def _key(text: str, template: str) -> str:
    return f"{settings.embedding_model}|{template}|{text}"


async def _embed_missing(pairs: list[tuple[str, str]]) -> dict[str, list[float]]:
    """(원문, 템플릿태그) 목록을 임베딩한다. 캐시에 없는 것만."""
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = [(t, tag) for t, tag in dict.fromkeys(pairs) if _key(t, tag) not in cache]
    if todo:
        print(f"  임베딩 {len(todo)}건 (캐시 적중 {len(set(pairs)) - len(todo)}건)")
        for i in range(0, len(todo), CHUNK):
            batch = todo[i : i + CHUNK]
            # 태그가 raw면 원문 그대로, norm-v1이면 정규화를 태운다
            vectors = await embedding.embed(
                [t if tag == "raw" else normalization.render(t) for t, tag in batch]
            )
            for (text, tag), vec in zip(batch, vectors, strict=True):
                cache[_key(text, tag)] = vec
            CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return cache


async def normalization_effect(rows: list[dict]) -> None:
    """정규화가 유사도를 얼마나 움직이는가."""
    print("\n" + "=" * 78)
    print("1. 정규화 전후 — 표기 차이가 유사도에 얼마나 섞이는가")
    print("=" * 78)

    # 정규화가 실제로 문자열을 바꾸는 것만 본다. 안 바뀌는 항목은 비교가 무의미하다
    targets = [
        r for r in rows
        if normalization.render(r["answer"]) != r["answer"]
        and normalization.render(r["answer"])  # 빈 문자열은 임베딩 불가
    ]
    print(f"  대상: {len(targets)}건 (정규화가 문자열을 바꾸고, 결과가 비어 있지 않은 것)")

    cache = await _embed_missing(
        [(r["answer"], "raw") for r in targets] + [(r["answer"], "norm-v1") for r in targets]
    )

    deltas = []
    print(f"\n  {'답변':<22}{'원문':>9}{'정규화':>9}{'변화':>9}  라벨")
    print("  " + "-" * 62)
    for r in targets:
        raw_sim = (await vector_store.match(cache[_key(r["answer"], "raw")], r["question_id"])).similarity
        norm_sim = (await vector_store.match(cache[_key(r["answer"], "norm-v1")], r["question_id"])).similarity
        if raw_sim is None or norm_sim is None:
            continue
        deltas.append((norm_sim - raw_sim, r["label"], r["answer"], raw_sim, norm_sim))

    for d, label, ans, raw, norm in sorted(deltas, key=lambda x: -abs(x[0]))[:10]:
        print(f"  {ans[:20]:<22}{raw:>9.3f}{norm:>9.3f}{d:>+9.3f}  {label}")

    up = [d for d, label, *_ in deltas if label == "correct"]
    down = [d for d, label, *_ in deltas if label == "incorrect"]
    print(f"\n  정답 {len(up)}건 평균 변화: {stat.mean(up):+.4f}" if up else "")
    print(f"  오답 {len(down)}건 평균 변화: {stat.mean(down):+.4f}" if down else "")
    print(f"  전체 평균 |변화|: {stat.mean(abs(d) for d, *_ in deltas):.4f}")
    print("\n  읽는 법: 정규화의 목적은 유사도를 올리는 것이 아니라")
    print("           **표기 차이로 생기는 흔들림을 없애는 것**이다.")
    print("           정답이 오르고 오답이 안 오르면 의도대로 작동한 것이다.")


async def paraphrase_invariance(rows: list[dict]) -> None:
    """같은 뜻을 다르게 쓴 답변들이 비슷한 유사도를 받는가."""
    print("\n" + "=" * 78)
    print("2. 패러프레이즈 불변성 — 뜻이 같으면 값도 비슷해야 한다")
    print("=" * 78)

    groups: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("pair_id") and r["label"] == "correct":
            groups.setdefault(r["pair_id"], []).append(r)
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    print(f"  대상: 쌍 {len(groups)}개 (정답 라벨만, 2건 이상인 것)")

    measured = json.loads((ROOT / "eval" / "measurements.json").read_text(encoding="utf-8"))
    sim_by_id = {m["id"]: m["similarity"] for m in measured if m.get("similarity") is not None}

    spreads = []
    print(f"\n  {'쌍':<16}{'건수':>5}{'최소':>9}{'최대':>9}{'폭':>9}")
    print("  " + "-" * 52)
    for pid, members in sorted(groups.items()):
        sims = [sim_by_id[m["id"]] for m in members if m["id"] in sim_by_id]
        if len(sims) < 2:
            continue
        spread = max(sims) - min(sims)
        spreads.append((spread, pid, members, sims))
    for spread, pid, members, sims in sorted(spreads, key=lambda x: -x[0])[:8]:
        print(f"  {pid:<16}{len(sims):>5}{min(sims):>9.3f}{max(sims):>9.3f}{spread:>9.3f}")

    if spreads:
        all_spreads = [s for s, *_ in spreads]
        print(f"\n  폭 중앙값 {stat.median(all_spreads):.3f} · 최대 {max(all_spreads):.3f}")
        print("\n  가장 벌어진 쌍의 내용")
        worst_spread, worst_pid, worst_members, _ = max(spreads, key=lambda x: x[0])
        for m in worst_members:
            s = sim_by_id.get(m["id"])
            if s is not None:
                print(f"    {s:.3f}  {m['answer'][:52]}")
        print("\n  읽는 법: 폭이 크다는 것은 **같은 뜻인데 값이 크게 다르다**는 뜻이고,")
        print("           그만큼 단일 임계값으로 가르기 어렵다는 신호다.")


async def main() -> None:
    rows = json.loads(DATASET.read_text(encoding="utf-8"))
    stored = await vector_store.count()
    if stored == 0:
        raise SystemExit("앵커가 없습니다. `uv run python -m scripts.seed`를 먼저 돌리세요")
    print(f"데이터셋 {len(rows)}건 · 앵커 {stored}건")
    await normalization_effect(rows)
    await paraphrase_invariance(rows)


if __name__ == "__main__":
    asyncio.run(main())
