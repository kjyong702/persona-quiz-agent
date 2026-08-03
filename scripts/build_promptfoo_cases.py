"""promptfoo 테스트 케이스를 데이터셋에서 만든다.

    uv run python -m scripts.build_promptfoo_cases

**왜 전체 372건이 아니라 LLM에 위임되는 것만 넣는가.**

운영에서 LLM 판정을 받는 답변은 임계값 사이에 떨어진 것들뿐이다. 전체를 재면
프롬프트 자체의 성능은 알 수 있지만 **이 시스템이 실제로 얼마나 맞히는지**는
안 나온다. 알고 싶은 것은 후자다. 임베딩이 확정한 몫과 합쳐야 시스템 정확도가
되기 때문이다.

그래서 `eval/measurements.json`의 유사도로 현재 임계값을 적용해 위임 대상만
고른다. 임계값을 바꾸면 대상도 바뀌므로 **임계값을 옮긴 뒤에는 다시 만들어야 한다.**
"""

import json
from pathlib import Path

import yaml

from app.core.config import settings
from app.core.negation import has_negation

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval" / "promptfoo" / "cases-judge.yaml"


def route(m: dict) -> str:
    """judge_service.judge와 같은 분기. 여기가 어긋나면 대상 선정이 틀린다."""
    sim = m.get("similarity")
    if sim is None:
        return "fallback"
    if sim >= settings.upper_threshold:
        return "llm" if has_negation(m["answer"]) else "embedding-correct"
    if sim <= settings.lower_threshold:
        return "embedding-incorrect"
    return "llm"


def main() -> None:
    dataset = {r["id"]: r for r in json.loads((ROOT / "eval" / "dataset.json").read_text("utf-8"))}
    measurements = json.loads((ROOT / "eval" / "measurements.json").read_text("utf-8"))
    # seed의 문항 순서가 곧 question_id다(1부터). 데이터셋의 question_text와
    # 대조해 어긋나면 즉시 멈춘다. 조용히 다른 문항의 정답으로 채점하면
    # 결과가 통째로 무의미해진다
    questions = [
        q
        for s in json.loads((ROOT / "seed" / "quiz-sets.json").read_text("utf-8"))
        for q in s["questions"]
    ]

    counts: dict[str, int] = {}
    tests = []
    for m in measurements:
        path = route(m)
        counts[path] = counts.get(path, 0) + 1
        if path not in ("llm", "fallback"):
            continue
        row = dataset[m["id"]]
        q = questions[row["question_id"] - 1]
        if q["question_text"] != row["question_text"]:
            raise SystemExit(
                f"question_id {row['question_id']}가 seed와 어긋납니다. "
                "데이터셋이나 seed 문항 순서가 바뀌었습니다"
            )
        tests.append({
            "description": f"{row['id']} [{row['category']}] {row['answer'][:34]}",
            "vars": {
                "question_text": q["question_text"],
                # promptfoo vars는 스칼라가 다루기 쉬워 파이프로 잇는다.
                # judge_prompt.py가 다시 쪼개 실제 호출과 같은 문자열을 만든다
                "expected_answers": " | ".join(q["expected_answers"]),
                "answer": row["answer"],
                "category": row["category"],
            },
            "assert": [{"type": "equals", "value": row["label"]}],
        })

    OUT.write_text(
        "# 이 파일은 scripts/build_promptfoo_cases.py가 만든다. 직접 고치지 말 것.\n"
        f"# 임계값 upper={settings.upper_threshold} lower={settings.lower_threshold} 기준.\n"
        + yaml.safe_dump(tests, allow_unicode=True, sort_keys=False, width=200),
        encoding="utf-8",
    )

    total = sum(counts.values())
    print(f"경로 분포 (전체 {total}건)")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<20} {v:>4}건 ({v/total:.1%})")
    print(f"\n케이스 {len(tests)}건 -> {OUT.relative_to(ROOT)}")
    print(f"판정 모델은 {settings.judge_model}. judge.yaml의 provider와 같아야 한다")


if __name__ == "__main__":
    main()
