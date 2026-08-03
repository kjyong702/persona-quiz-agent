"""흔들리는 판정은 모델이 헷갈리는 자리인가.

    uv run python -m scripts.logprob_probe

**가설.** temperature 0에서 결과가 갈리는 이유는 최상위 두 토큰의 확률이 붙어
있기 때문이다. 붙어 있으면 부동소수점 누적 순서가 조금만 달라져도 순위가 뒤집힌다.
반대로 확률이 확실히 벌어져 있으면 몇 번을 돌려도 같은 답이 나온다.

**검증 방법.** `logprobs`로 correct와 incorrect의 확률을 직접 받아 그 차이를
재고, 20회 반복에서 실제로 흔들린 케이스와 맞춰본다. 가설이 맞으면
**흔들린 케이스만 차이가 작아야 한다.**

맞으면 이건 관측 수단이 된다. 판정할 때마다 이 차이를 보고 붙어 있는 것만
따로 다룰 수 있다.
"""

import asyncio
import json
import math
from pathlib import Path

import yaml

from app.core import prompts
from app.core.config import settings
from app.core.llm import _get_client, build_user_message

ROOT = Path(__file__).resolve().parent.parent


async def probe(question_text: str, expected: list[str], answer: str) -> dict:
    r = await _get_client().chat.completions.create(
        model=settings.judge_model,
        messages=[
            {"role": "system", "content": prompts.load(prompts.JUDGE_PROMPT)},
            {"role": "user", "content": build_user_message(question_text, expected, answer)},
        ],
        temperature=0,
        max_tokens=5,
        logprobs=True,
        top_logprobs=5,
    )
    top = r.choices[0].logprobs.content[0].top_logprobs
    # 첫 토큰만 본다. correct/incorrect가 여기서 갈린다
    probs = {t.token.strip().lower(): math.exp(t.logprob) for t in top}
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    first, second = ranked[0], (ranked[1] if len(ranked) > 1 else ("", 0.0))
    return {
        "top": first[0], "p1": first[1],
        "second": second[0], "p2": second[1],
        "margin": first[1] - second[1],
    }


async def main() -> None:
    cases = yaml.safe_load((ROOT / "eval" / "promptfoo" / "cases-boundary.yaml").read_text("utf-8"))
    # 20회 반복에서 실제로 갈린 것 (docs/notes/determinism.md 참고)
    flaky = {
        "태양에서 다섯 번째로 떨어져 있는 가스로 된 행성이요",
        "조선 4대 임금이요",
        "365",
        "자석에 붙는 그 금속이요",
    }
    print(f"{'답변':<40}{'1위':>10}{'확률':>8}{'2위':>10}{'확률':>8}{'차이':>8}  20회")
    print("-" * 92)
    rows = []
    for c in cases:
        v = c["vars"]
        r = await probe(v["question_text"], [s.strip() for s in v["expected_answers"].split("|")], v["answer"])
        is_flaky = v["answer"] in flaky
        rows.append((is_flaky, r["margin"]))
        print(f"{v['answer'][:38]:<40}{r['top'][:8]:>10}{r['p1']:>8.3f}"
              f"{r['second'][:8]:>10}{r['p2']:>8.3f}{r['margin']:>8.3f}"
              f"  {'**갈림**' if is_flaky else '고정'}")

    fl = [m for f, m in rows if f]
    st = [m for f, m in rows if not f]
    print("\n" + "=" * 92)
    print(f"  20회에서 갈린 {len(fl)}건 : 확률 차이 중앙값 {sorted(fl)[len(fl)//2]:.3f}  (최대 {max(fl):.3f})")
    print(f"  20회 내내 고정 {len(st)}건 : 확률 차이 중앙값 {sorted(st)[len(st)//2]:.3f}  (최소 {min(st):.3f})")
    if max(fl) < min(st):
        print(f"\n  **완전히 갈린다.** 갈린 것은 전부 {max(fl):.3f} 이하, 고정된 것은 전부 {min(st):.3f} 이상")
    else:
        print(f"\n  겹치는 구간이 있다. 갈린 것의 최대 {max(fl):.3f} > 고정의 최소 {min(st):.3f}")


if __name__ == "__main__":
    asyncio.run(main())
