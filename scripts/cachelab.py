"""프롬프트 캐싱이 실제로 걸리는지, 무엇이 캐싱을 죽이는지 재는 실험.

    uv run python -m scripts.cachelab

**왜 이 실험이 필요한가**

`persona_prompt.py`에 "프리픽스에 변동 값을 넣지 마라"고 주석으로 적어뒀다.
그런데 그건 **읽으면 당연하고 지키기는 쉬운 규칙**이라 오히려 위험하다.
어기면 얼마나 나빠지는지를 모르면, 나중에 누가 "세션 ID 하나쯤" 하고 넣는다.

그래서 일부러 어겨보고 숫자로 남긴다.

**세 조건**

    A 정상      [ 고정 프리픽스 ][ 페르소나 델타 ]
    B 오염      [ 세션 ID ][ 고정 프리픽스 ][ 페르소나 델타 ]
    C 순서뒤집기 [ 페르소나 델타 ][ 고정 프리픽스 ]

캐싱은 **문장 맨 앞부터** 일치하는 구간에만 걸린다. B는 맨 앞이 매번 다르고,
C는 페르소나가 바뀔 때마다 맨 앞이 달라진다. 둘 다 A와 내용은 같지만
배치만 다르다. **같은 정보를 어디에 두느냐가 비용을 가른다는 것을 보여준다.**
"""

import asyncio
import json
import time
from pathlib import Path

import httpx

from app.core import persona_prompt, prompts
from app.core.config import settings

ROOT = Path(__file__).resolve().parent.parent
CALLS_PER_PERSONA = 3  # 첫 호출은 캐시를 채우는 데 쓰이므로 여러 번 돌린다


class _Persona:
    """DB 없이 시드 JSON으로 페르소나를 흉내낸다. 실험에 DB는 필요 없다."""

    def __init__(self, raw: dict[str, str]) -> None:
        self.name = raw["name"]
        self.personality = raw["personality"]
        self.speech_style = raw["speech_style"]
        self.reaction_style = raw["reaction_style"]


def _delta(p: _Persona) -> str:
    return (
        "# 이번 진행자\n\n"
        f"이름: {p.name}\n성격: {p.personality}\n"
        f"말투: {p.speech_style}\n리액션: {p.reaction_style}"
    )


def build(condition: str, p: _Persona, session_id: int) -> str:
    base = prompts.load(persona_prompt.BASE_PROMPT)
    if condition == "A":
        return f"{base}\n\n{_delta(p)}"
    if condition == "B":
        # 맨 앞에 매번 달라지는 값을 넣는다. 이것이 캐싱을 죽이는 전형적 실수다
        return f"세션 식별자: {session_id}\n\n{base}\n\n{_delta(p)}"
    if condition == "C":
        # 내용은 A와 같고 순서만 뒤집었다
        return f"{_delta(p)}\n\n{base}"
    raise ValueError(condition)


async def _call(client: httpx.AsyncClient, system: str, user: str) -> tuple[int, int]:
    r = await client.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {settings.openai_api_key}"},
        json={
            "model": settings.judge_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": settings.host_temperature,
            "max_tokens": 60,
        },
    )
    usage = r.json()["usage"]
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    return usage["prompt_tokens"], cached


async def run(client: httpx.AsyncClient, condition: str, personas: list[_Persona]) -> dict:
    total_in = total_cached = calls = 0
    session_id = 1000
    # 페르소나를 번갈아 쓴다. 실제 서비스에서 세션마다 진행자가 다른 상황이다
    for i in range(CALLS_PER_PERSONA):
        for p in personas:
            session_id += 1
            system = build(condition, p, session_id)
            user = persona_prompt.build_user_message(
                persona_prompt.QUESTION,
                question_text="대한민국의 수도는 어디인가요?",
                order_no=i + 1,
                total=10,
            )
            pt, ct = await _call(client, system, user)
            total_in += pt
            total_cached += ct
            calls += 1
    return {
        "condition": condition,
        "calls": calls,
        "input_tokens": total_in,
        "cached_tokens": total_cached,
        "hit_rate": total_cached / total_in if total_in else 0.0,
    }


def _cost(input_tokens: int, cached: int) -> float:
    """캐시된 토큰은 절반 값이다 (gpt-4o-mini 기준)."""
    uncached = input_tokens - cached
    return (uncached * 0.15 + cached * 0.075) / 1e6


async def main() -> None:
    raw = json.loads((ROOT / "seed" / "personas.json").read_text(encoding="utf-8"))
    personas = [_Persona(p) for p in raw]

    labels = {
        "A": "정상      [프리픽스][델타]",
        "B": "오염      [세션ID][프리픽스][델타]",
        "C": "순서뒤집기 [델타][프리픽스]",
    }
    print(f"페르소나 {len(personas)}종을 번갈아 {CALLS_PER_PERSONA}회씩 호출\n")
    print(f"{'조건':<32}{'호출':>5}{'입력토큰':>10}{'캐시적중':>10}{'적중률':>8}{'비용':>12}")
    print("-" * 80)

    results = []
    async with httpx.AsyncClient(timeout=60) as client:
        for cond in ("A", "B", "C"):
            r = await run(client, cond, personas)
            r["cost"] = _cost(r["input_tokens"], r["cached_tokens"])
            results.append(r)
            print(
                f"{labels[cond]:<32}{r['calls']:>5}{r['input_tokens']:>10,}"
                f"{r['cached_tokens']:>10,}{r['hit_rate']:>7.1%}"
                f"${r['cost']:>10.4f}"
            )

    a, b, c = results
    print("\n해석")
    if a["cached_tokens"] > 0:
        print(f"  A는 적중률 {a['hit_rate']:.1%}. 프리픽스를 앞에 고정하면 재사용된다")
    else:
        print("  ⚠️ A조차 적중이 0이다. 프리픽스가 1,024토큰에 못 미치는지 확인할 것")
    for other, name in ((b, "B 오염"), (c, "C 순서뒤집기")):
        if a["cached_tokens"] > 0 and other["cached_tokens"] < a["cached_tokens"]:
            lost = 1 - (other["cached_tokens"] / a["cached_tokens"])
            up = (other["cost"] / a["cost"] - 1) if a["cost"] else 0
            print(f"  {name}: 적중 {lost:.0%} 손실, 같은 호출에 비용 {up:+.1%}")
    print("\n  세 조건의 프롬프트 내용은 같다. 배치만 다르다.")


def _synthetic(n: int, nonce: str) -> list[_Persona]:
    """페르소나를 n종으로 늘린다. 개인화 대상이 많아지는 상황을 흉내낸다.

    **nonce가 중요하다.** 이름에 실행마다 다른 값을 넣지 않으면 앞선 실행이
    올려둔 캐시가 남아 있어 측정이 오염된다. 실제로 처음 돌렸을 때 그 때문에
    C가 A보다 좋게 나오는 엉뚱한 결과가 나왔다.

    반대로 **공통 프리픽스는 일부러 그대로 둔다.** 실서비스에서 프리픽스는
    계속 따뜻하게 유지되는 것이 정상이고, 그게 이 배치의 이점이기 때문이다.
    """
    base = json.loads((ROOT / "seed" / "personas.json").read_text(encoding="utf-8"))
    out = []
    for i in range(n):
        src = base[i % len(base)]
        out.append(_Persona({
            "name": f"{src['name']}{nonce}{i}",
            "personality": f"{src['personality']} 진행자 식별 코드는 {nonce}{i}이다",
            "speech_style": src["speech_style"],
            "reaction_style": src["reaction_style"],
        }))
    return out


async def scale_test() -> None:
    """페르소나 수를 늘리며 A와 C의 적중률이 어떻게 갈리는지 본다.

    **이것이 이 실험의 본체다.** 페르소나가 두 종뿐이면 C도 각자 캐시를 만들어
    A와 차이가 안 난다. 개인화 대상이 늘어나야 배치의 차이가 드러난다.

    A는 프리픽스가 맨 앞이라 **캐시 엔트리 하나를 모두가 공유한다.**
    C는 맨 앞이 페르소나마다 달라 **엔트리가 페르소나 수만큼 필요하다.**
    """
    print("\n\n" + "=" * 80)
    print("페르소나 수에 따른 적중률 (결론 미도출 — 아래 경고 참고)")
    print("=" * 80)
    print(f"{'페르소나 수':>10}{'A 적중률':>12}{'C 적중률':>12}{'차이':>10}")
    print("-" * 80)
    nonce = f"{int(time.time()) % 100000:05d}"
    print(f"  (실행 식별자 {nonce} — 앞선 실행의 캐시와 겹치지 않게 한다)\n")
    async with httpx.AsyncClient(timeout=60) as client:
        for n in (2, 10):
            personas = _synthetic(n, nonce)
            a = await run(client, "A", personas)
            c = await run(client, "C", personas)
            print(f"{n:>10}{a['hit_rate']:>11.1%}{c['hit_rate']:>11.1%}"
                  f"{a['hit_rate']-c['hit_rate']:>+9.1%}")
    print("\n  ⚠️ 이 비교는 결론을 내지 못했다. 실행마다 적중률이 29%에서 90%까지")
    print("     흔들린다. OpenAI 자동 캐싱은 보장이 아니라 최선 노력이라 같은 조건에서도")
    print("     값이 크게 달라진다. A와 C의 차이를 재려면 표본이 훨씬 커야 한다.")
    print("\n  확실한 것은 B(프리픽스 앞에 변동 값)뿐이다. 그건 매번 0이 나온다.")


async def ab_test() -> None:
    """A와 C를 제대로 갈라내는 재실험.

    **앞선 실험이 결론을 못 낸 이유 두 가지를 고쳤다.**

    1. **표본 부족.** 페르소나 10종에 3회씩이면 이론 격차가 30%p인데
       최선 노력 캐싱의 편차에 묻혔다. 20종에 2회씩으로 바꾸면 이론상
       A 97% 대 C 50%로 격차가 47%p가 되어 노이즈를 넘는다
    2. **시점이 달랐다.** A를 다 돌리고 C를 돌리면 서로 다른 시간대에
       측정된다. 캐시 상태가 시간에 따라 변하면 그게 조건 차이로 잡힌다.
       **두 조건을 번갈아 호출해** 같은 시간대를 겪게 한다

    이론 예측 (페르소나 N종, 각 M회)

        A: 맨 처음 한 번만 미스        -> (N*M - 1) / (N*M)
        C: 페르소나마다 첫 호출 미스    -> (M - 1) / M
    """
    N, M = 20, 2
    nonce = f"{int(time.time()) % 100000:05d}"
    personas = _synthetic(N, nonce)

    print("\n\n" + "=" * 80)
    print(f"재실험 — A와 C를 번갈아 호출 (페르소나 {N}종 x {M}회, 실행 식별자 {nonce})")
    print("=" * 80)
    print(f"  이론 예측:  A {(N*M-1)/(N*M):.0%}   C {(M-1)/M:.0%}")
    print()

    totals = {"A": [0, 0], "C": [0, 0]}  # [입력, 적중]
    session_id = 9000
    async with httpx.AsyncClient(timeout=60) as client:
        for rnd in range(M):
            for p in personas:
                for cond in ("A", "C"):
                    session_id += 1
                    pt, ct = await _call(
                        client,
                        build(cond, p, session_id),
                        persona_prompt.build_user_message(
                            persona_prompt.QUESTION,
                            question_text="대한민국의 수도는 어디인가요?",
                            order_no=rnd + 1,
                            total=10,
                        ),
                    )
                    totals[cond][0] += pt
                    totals[cond][1] += ct

    print(f"{'조건':<28}{'호출':>6}{'입력토큰':>11}{'캐시적중':>11}{'적중률':>9}")
    print("-" * 80)
    for cond, label in (("A", "A 정상   [프리픽스][델타]"), ("C", "C 뒤집기 [델타][프리픽스]")):
        tin, tc = totals[cond]
        print(f"{label:<28}{N*M:>6}{tin:>11,}{tc:>11,}{tc/tin:>8.1%}")

    ra = totals["A"][1] / totals["A"][0]
    rc = totals["C"][1] / totals["C"][0]
    print(f"\n  실측 격차: {ra-rc:+.1%}p")
    if ra - rc > 0.15:
        print("  -> 가설대로다. 프리픽스를 앞에 두면 캐시 엔트리 하나를 전부가 공유한다.")
        print("     뒤집으면 페르소나마다 엔트리가 따로 필요해 첫 호출이 매번 미스다.")
    elif abs(ra - rc) <= 0.15:
        print("  -> 격차가 이론 예측(47%p)에 한참 못 미친다. 가설이 틀렸거나")
        print("     최선 노력 캐싱의 편차가 여전히 지배적이다.")
    else:
        print("  -> C가 오히려 낫다. 캐싱 동작에 대한 이해를 다시 봐야 한다.")


if __name__ == "__main__":
    asyncio.run(ab_test())
