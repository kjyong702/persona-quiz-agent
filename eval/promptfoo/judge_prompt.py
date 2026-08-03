"""promptfoo가 판정 프롬프트를 만들 때 부르는 함수.

**프롬프트를 여기에 복사하지 않는다.** `app.core`를 그대로 부른다.

평가용으로 프롬프트를 베껴두면 운영 프롬프트를 고쳤을 때 평가가 옛 버전을
계속 재고, 그 사실을 아무도 모른다. 평가가 통과하는데 서비스는 달라져 있는
상태가 가장 나쁘다. 그래서 `build_user_message`와 `prompts.load`를 직접 부른다.

promptfoo가 이 파일을 부를 때 저장소 루트가 sys.path에 없으므로 직접 넣는다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.core import prompts  # noqa: E402
from app.core.llm import build_user_message  # noqa: E402


def _messages(version: str, context: dict) -> list[dict]:
    v = context["vars"]
    expected = v["expected_answers"]
    if isinstance(expected, str):
        expected = [s.strip() for s in expected.split("|")]
    return [
        {"role": "system", "content": prompts.load(version)},
        {
            "role": "user",
            "content": build_user_message(v["question_text"], expected, v["answer"]),
        },
    ]


def judge(context: dict) -> list[dict]:
    """운영이 지금 쓰는 버전. 상수를 참조하므로 운영이 바뀌면 여기도 따라온다."""
    return _messages(prompts.JUDGE_PROMPT, context)


def judge_v2(context: dict) -> list[dict]:
    """후보 버전. 두 함수의 차이가 프롬프트 파일 하나뿐이어야 비교가 성립한다."""
    return _messages("judge.v2", context)


def judge_v3(context: dict) -> list[dict]:
    """v2가 새로 깨뜨린 3건을 겨냥한 수정본. 무엇을 고쳤는지는 프롬프트 diff에 있다."""
    return _messages("judge.v3", context)
