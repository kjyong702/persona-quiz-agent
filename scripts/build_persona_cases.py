"""페르소나 회귀 케이스를 만든다.

    uv run python -m scripts.build_persona_cases

**무엇을 지키는지 정하고 그것만 본다.** 멘트가 좋은지는 채점할 수 없다.
정답이 없기 때문이다. 대신 무너지면 페르소나가 무의미해지는 규칙 둘을 본다.

1. **출제 멘트에 기대 정답이 들어가면 안 된다.** 문제 문장을 넘겨야 하므로
   모델이 답을 알고 있고, 지시문으로만 막고 있는 유일한 자리다
2. **페르소나마다 말투가 갈려야 한다.** 불꽃은 반말과 느낌표, 고요는 존댓말과
   느낌표 없음. 설정만 써두고 지켜지는지 안 보면 페르소나는 장식이다

**규칙 검사는 정규식으로 한다.** 말투를 LLM에게 채점시키면 그 채점도 흔들린다
(`docs/notes/determinism.md`). 정규식으로 되는 것을 굳이 LLM에 맡기지 않는다.
"""

import json
from pathlib import Path

import yaml

from app.core import persona_prompt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval" / "promptfoo" / "cases-persona.yaml"

# 존댓말 어미. 문장 끝이 아니어도 나타나므로 위치 조건을 걸지 않는다
POLITE = r"(습니다|입니다|세요|해요|네요|십니다|시죠|십시오)"

# 반말 어미는 **문장 끝에서만** 어미다. 중간에서는 다른 형태소라서
# 위치를 조건에 넣지 않으면 안 잡힌다. 처음에 `(야|어|지|자|해)\s`로 썼다가
# 두 페르소나 모두 0회가 나왔다. 상세는 docs/notes/persona-regression.md
CASUAL = r"[어아야지자해네](?=[!?.]|$)"

STYLE = {
    "불꽃": [
        {"type": "javascript", "value": f"/{POLITE}/.test(output) === false",
         "metric": "존댓말-없음"},
        {"type": "javascript", "value": "output.includes('!')", "metric": "느낌표-있음"},
    ],
    "고요": [
        {"type": "javascript", "value": f"/{POLITE}/.test(output)", "metric": "존댓말-있음"},
        {"type": "javascript", "value": "output.includes('!') === false", "metric": "느낌표-없음"},
    ],
}


def main() -> None:
    personas = json.loads((ROOT / "seed" / "personas.json").read_text("utf-8"))
    questions = [
        q
        for s in json.loads((ROOT / "seed" / "quiz-sets.json").read_text("utf-8"))
        for q in s["questions"]
    ][:4]

    tests = []
    for persona in personas:
        base = {
            "persona_name": persona["name"],
            "personality": persona["personality"],
            "speech_style": persona["speech_style"],
            "reaction_style": persona["reaction_style"],
        }
        style = STYLE[persona["name"]]

        # 출제 — 문제 문장을 넘기는 유일한 자리라 유출 검사가 여기 붙는다
        for q in questions:
            leak = [
                # 한 글자 앵커(원, 철)는 다른 낱말에 우연히 들어가므로 뺀다
                {"type": "not-icontains", "value": token, "metric": "정답-유출"}
                for token in q["expected_answers"] if len(token.strip()) >= 2
            ]
            tests.append({
                "description": f"{persona['name']}/출제/{q['question_text'][:20]}",
                "vars": {**base, "situation": persona_prompt.QUESTION,
                         "question_text": q["question_text"], "order_no": 1, "total": 10},
                "assert": leak + style,
            })

        # 나머지 상황 — 말투만 본다. 문제도 정답도 안 넘기므로 유출은 구조적으로 불가능
        for situation, extra in (
            (persona_prompt.OPENING, {"total": 10}),
            (persona_prompt.CORRECT, {"order_no": 2, "total": 10}),
            (persona_prompt.INCORRECT, {"order_no": 3, "total": 10}),
            (persona_prompt.CLOSING, {"score": 7, "total": 10}),
        ):
            tests.append({
                "description": f"{persona['name']}/{situation}",
                "vars": {**base, "situation": situation, **extra},
                "assert": list(style),
            })

    OUT.write_text(
        "# scripts/build_persona_cases.py가 만든다. 직접 고치지 말 것.\n"
        + yaml.safe_dump(tests, allow_unicode=True, sort_keys=False, width=200),
        encoding="utf-8",
    )
    leak_checks = sum(1 for t in tests for a in t["assert"] if a["metric"] == "정답-유출")
    print(f"케이스 {len(tests)}건 (페르소나 {len(personas)} × 상황 {len(questions) + 4})")
    print(f"  정답 유출 검사 {leak_checks}개 · 말투 검사 {len(tests) * 2}개")
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
