"""부정 표현 감지 테스트.

**여기서 진짜로 지키는 것은 두 가지다.**

1. 한국어 활용형을 놓치지 않는가. `아닙니다`는 문자열 `아니`를 포함하지 않는다
2. 기대 정답에 쓰이는 부정 표현(`없`, `않`)에 오작동하지 않는가

2번이 특히 중요하다. 14번 문항은 질문 자체가 부정문이고 정답도 부정 표현이라,
규칙을 넓게 잡으면 정답이 걸린다.
"""

import pytest

from app.core.negation import has_negation

# 평가셋에서 실제로 상한을 넘긴 부정문 오답들. 여기 있는 것은 전부 잡아야 한다
REAL_NEGATIONS = [
    "섭씨 0도는 아니에요",
    "왓슨과 크릭 아니에요",
    "빨강 초록 파랑 아니에요",
    "4년마다가 아니에요",
    "왓슨이랑 크릭은 아니라던데요",
    "아인슈타인 아닙니다",  # 받침 때문에 "아니"로는 안 잡힌다
    "세종대왕 아닙니다",
    "적혈구 아니에요",
    "진공 아닙니다",
    "에베레스트산이 아니라던데요",
    "0도 아닙니다",
    "진동수 아닙니다",
    "진공 상태는 아닌 것 같은데요",  # "아닌"도 받침 형태다
    "서울 아냐",
    "목성 아님",
]

# 14번 문항(소리는 어떤 환경에서 전달되지 않나요?)의 기대 정답과 정답 변형들.
# 부정 표현을 쓰지만 **정답**이므로 걸리면 안 된다
CORRECT_WITH_NEGATIVE_WORDS = [
    "공기가 없는 곳에서요",
    "진공 상태에서는 전달이 안 돼요",
    "매질이 하나도 없는 우주 공간 같은 데요",
    "진공에서는 전달되지 않습니다",
    "소리는 매질이 있어야 퍼지는 파동이라서 진공에서는 전달이 안 된다고 배웠어요",
]

PLAIN_ANSWERS = ["서울", "목성", "세종대왕", "4년마다", "88개", "몰라요", "RGB"]


@pytest.mark.parametrize("text", REAL_NEGATIONS)
def test_활용형을_전부_잡는다(text: str) -> None:
    """어간 아니-의 활용형은 받침 때문에 음절이 달라진다.

    아니에요 / 아닙니다 / 아닌 / 아냐 / 아님을 모두 잡아야 한다.
    """
    assert has_negation(text), f"놓쳤다: {text!r}"


@pytest.mark.parametrize("text", CORRECT_WITH_NEGATIVE_WORDS)
def test_정답의_부정_표현에는_반응하지_않는다(text: str) -> None:
    """`없`, `않`, `안`은 규칙에서 일부러 뺐다.

    14번 문항은 질문이 부정문이라 정답에도 부정 표현이 들어간다.
    여기서 발동하면 정답을 매번 LLM으로 보내 비용만 늘어난다.
    """
    assert not has_negation(text), f"오작동: {text!r}"


@pytest.mark.parametrize("text", PLAIN_ANSWERS)
def test_평범한_답변에는_반응하지_않는다(text: str) -> None:
    assert not has_negation(text)


def test_빈_문자열도_안전하다() -> None:
    assert not has_negation("")
