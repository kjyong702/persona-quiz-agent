"""페르소나 진행 멘트를 실제로 뽑아 두 가지를 본다.

    uv run python -m scripts.personalab

**1. 정답 유출** — 이것이 주 목적이다.

프롬프트에 "정답을 흘리지 마라"를 여러 번 적었고, 리액션 생성에는 문제 문장과
기대 정답을 아예 넘기지 않았다. **그런데 지킨다는 것을 확인한 적이 없다.**

특히 위험한 자리는 출제(question)다. 문제 문장은 넘겨야 하는데, 모델이
"대한민국의 수도는 어디인가요?"를 보고 스스로 서울을 알기 때문에
진행 멘트에 흘릴 수 있다. **프롬프트 지시만으로 막고 있는 유일한 자리다.**

**2. 말투 대비** — 페르소나 설정이 실제로 출력을 가르는지.

설정에 "반말, 느낌표 많이"와 "존댓말, 느낌표 없이"를 넣었는데, 서술만으로는
모델이 잘 안 따른다는 것이 알려져 있다. 눈으로 확인한다.
"""

import asyncio
import json
import re
from pathlib import Path

from app.core import persona_prompt
from app.core.llm import generate_host_message

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_PER_SITUATION = 2


class _Persona:
    def __init__(self, raw: dict[str, str]) -> None:
        self.name = raw["name"]
        self.personality = raw["personality"]
        self.speech_style = raw["speech_style"]
        self.reaction_style = raw["reaction_style"]


def _leaks(text: str, expected: list[str]) -> list[str]:
    """멘트에 기대 정답이 들어 있는가.

    짧은 앵커(`원`, `철`)는 다른 단어에 우연히 들어가므로 두 글자 이상만 본다.
    영문 앵커는 대소문자를 무시한다.
    """
    found = []
    for answer in expected:
        token = answer.strip()
        if len(token) < 2:
            continue
        if re.search(re.escape(token), text, re.IGNORECASE):
            found.append(token)
    return found


async def main() -> None:
    personas = [
        _Persona(p)
        for p in json.loads((ROOT / "seed" / "personas.json").read_text(encoding="utf-8"))
    ]
    sets = json.loads((ROOT / "seed" / "quiz-sets.json").read_text(encoding="utf-8"))
    questions = sets[0]["questions"][:3]

    total = leaked = 0
    leak_samples: list[tuple[str, str, list[str], str]] = []

    print("=" * 78)
    print("1. 정답 유출 검사")
    print("=" * 78)

    for persona in personas:
        system = persona_prompt.build_system_prompt(persona)
        for q in questions:
            for _ in range(SAMPLES_PER_SITUATION):
                # 출제 멘트 — 문제 문장을 넘기므로 유출 위험이 가장 큰 자리
                msg = await generate_host_message(
                    system,
                    persona_prompt.build_user_message(
                        persona_prompt.QUESTION,
                        question_text=q["question_text"],
                        order_no=1,
                        total=10,
                    ),
                )
                total += 1
                hits = _leaks(msg, q["expected_answers"])
                if hits:
                    leaked += 1
                    leak_samples.append((persona.name, "question", hits, msg))

            # 오답 리액션 — 문제도 정답도 안 넘긴다. 여기서 새면 설계가 틀린 것
            msg = await generate_host_message(
                system,
                persona_prompt.build_user_message(
                    persona_prompt.INCORRECT, order_no=1, total=10
                ),
            )
            total += 1
            hits = _leaks(msg, q["expected_answers"])
            if hits:
                leaked += 1
                leak_samples.append((persona.name, "incorrect", hits, msg))

    print(f"  검사 {total}건 중 유출 {leaked}건 ({leaked/total:.0%})")
    if leak_samples:
        print("\n  유출 사례")
        for name, situation, hits, msg in leak_samples[:6]:
            print(f"    [{name}/{situation}] {hits} -> {msg[:60]}")
    else:
        print("  ✅ 유출 없음")

    print("\n" + "=" * 78)
    print("2. 말투 대비 — 같은 상황, 다른 페르소나")
    print("=" * 78)

    situations = [
        (persona_prompt.OPENING, {"total": 10}),
        (persona_prompt.QUESTION, {
            "question_text": questions[0]["question_text"], "order_no": 1, "total": 10}),
        (persona_prompt.CORRECT, {"order_no": 1, "total": 10}),
        (persona_prompt.INCORRECT, {"order_no": 2, "total": 10}),
        (persona_prompt.CLOSING, {"score": 7, "total": 10}),
    ]
    for situation, kwargs in situations:
        print(f"\n  [{situation}]")
        for persona in personas:
            msg = await generate_host_message(
                persona_prompt.build_system_prompt(persona),
                persona_prompt.build_user_message(situation, **kwargs),
            )
            print(f"    {persona.name}: {msg}")

    print("\n" + "=" * 78)
    print("3. 말투 지표 (설정이 출력을 실제로 가르는가)")
    print("=" * 78)
    for persona in personas:
        system = persona_prompt.build_system_prompt(persona)
        texts = []
        for situation, kwargs in situations:
            texts.append(
                await generate_host_message(
                    system, persona_prompt.build_user_message(situation, **kwargs)
                )
            )
        joined = " ".join(texts)
        bangs = joined.count("!")
        polite = len(re.findall(r"(습니다|입니다|세요|해요|네요|십니다)", joined))
        # 문장 끝의 반말 어미. 처음에는 어미를 단순 나열했다가 0회가 나왔다.
        # 한국어는 어미가 문장 끝에서만 어미이고 중간에서는 다른 형태소라,
        # **위치를 조건에 넣지 않으면 잡히지 않는다**
        casual = len(re.findall(r"[어아야지자해네](?=[!?.]|$)", joined))
        print(f"  {persona.name:<6} 느낌표 {bangs:>3}회  존댓말 어미 {polite:>3}회  "
              f"반말 신호 {casual:>3}회")
    print("\n  설정: 불꽃=반말/느낌표 많이, 고요=존댓말/느낌표 없이")


if __name__ == "__main__":
    asyncio.run(main())
