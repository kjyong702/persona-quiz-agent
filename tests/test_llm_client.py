"""LLM 판정 응답 파싱 테스트.

외부 호출 없이 순수 함수만 본다. 실제 호출 품질은 Phase 5 promptfoo가 맡는다.
"""

import pytest

from app.core import llm
from app.core.exceptions import LLMUnavailableError


@pytest.mark.parametrize(
    "raw",
    ["correct", "Correct", "CORRECT", " correct ", "correct.", "`correct`"],
)
def test_parse_verdict_correct(raw: str) -> None:
    assert llm.parse_verdict(raw) is True


@pytest.mark.parametrize(
    "raw",
    ["incorrect", "Incorrect", "INCORRECT", " incorrect ", "incorrect."],
)
def test_parse_verdict_incorrect(raw: str) -> None:
    """incorrect가 correct를 문자열로 포함한다.

    포함 검사로 짜거나 correct를 먼저 보면 오답이 정답으로 뒤집힌다.
    판정 파이프라인에서 가장 조용히 망가지는 자리라 따로 못박아 둔다.
    """
    assert llm.parse_verdict(raw) is False


@pytest.mark.parametrize("raw", ["", "글쎄요", "정답입니다", "yes", "1"])
def test_parse_verdict_rejects_unparseable(raw: str) -> None:
    """해석할 수 없으면 추측하지 않고 실패시킨다.

    임의로 오답 처리하면 틀린 판정이 평가 데이터에 그대로 남는다.
    """
    with pytest.raises(LLMUnavailableError):
        llm.parse_verdict(raw)


def test_build_user_message_includes_all_parts() -> None:
    message = llm.build_user_message(
        "대한민국의 수도는?", ["서울", "서울특별시"], "서울이요"
    )

    assert "대한민국의 수도는?" in message
    assert "서울, 서울특별시" in message
    assert "서울이요" in message
