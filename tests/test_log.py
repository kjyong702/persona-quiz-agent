"""로깅.

**여기서 확인하는 것은 "조용히 삼킨 것이 흔적을 남기는가"다.**

폴백은 옳은 동작이다. 임베딩이 죽어도 퀴즈는 LLM으로 계속 돌아야 한다.
문제는 그게 밖에서 보면 아무 일도 없어 보인다는 것이다. 나중에
"왜 그날 LLM 비용이 두 배였지"를 물으면 답할 근거가 없다.

`structlog.testing.capture_logs`로 실제 로그를 받아 본다. 로거를 목으로 바꾸면
프로세서 파이프라인을 타지 않아 **요청 ID가 붙는지 같은 것을 확인할 수 없다.**
"""

import pytest
import structlog

from app.core import embedding, llm, log, normalization, vector_store
from app.core.exceptions import EmbeddingUnavailableError
from app.models import JudgeMethod, Question
from app.services import judge_service


def _question() -> Question:
    return Question(
        id=1,
        quiz_set_id=1,
        order_no=1,
        question_text="대한민국의 수도는 어디인가요?",
        expected_answers=["서울"],
    )


def _events(captured: list[dict], name: str) -> list[dict]:
    return [entry for entry in captured if entry.get("event") == name]


@pytest.mark.asyncio
async def test_임베딩이_죽으면_폴백이_흔적을_남긴다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**이 테스트가 이 기능의 이유다.**

    예전에는 이 경로가 조용했다. 판정은 나가는데 임베딩이 죽었다는 사실이
    어디에도 안 남았다.
    """

    async def dead(text: str) -> list[float]:
        raise EmbeddingUnavailableError("임베딩 제공사 장애")

    async def ok(*args: object, **kwargs: object) -> bool:
        return True

    monkeypatch.setattr(embedding, "embed_one", dead)
    monkeypatch.setattr(llm, "judge_answer", ok)

    with structlog.testing.capture_logs() as captured:
        result = await judge_service.judge(_question(), "서울")

    assert result.judge_method == JudgeMethod.FALLBACK
    fallback = _events(captured, "judge.fallback")
    assert len(fallback) == 1
    assert fallback[0]["reason"] == "embedding_unavailable"
    assert fallback[0]["log_level"] == "warning"


@pytest.mark.asyncio
async def test_앵커가_없으면_설정_실수일_수_있어_남긴다(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_anchor(vector: list[float], question_id: int) -> vector_store.AnchorMatch:
        return vector_store.AnchorMatch(similarity=None, rival_similarity=None)

    async def embed(text: str) -> list[float]:
        return [1.0, 0.0]

    async def ok(*args: object, **kwargs: object) -> bool:
        return True

    monkeypatch.setattr(embedding, "embed_one", embed)
    monkeypatch.setattr(vector_store, "match", no_anchor)
    monkeypatch.setattr(llm, "judge_answer", ok)

    with structlog.testing.capture_logs() as captured:
        await judge_service.judge(_question(), "서울")

    fallback = _events(captured, "judge.fallback")
    assert fallback[0]["reason"] == "no_anchor"


@pytest.mark.asyncio
async def test_정규화가_비우는_답변도_남긴다(monkeypatch: pytest.MonkeyPatch) -> None:
    """제공사는 멀쩡한데 우리가 못 보낼 입력을 보낸 경우다.

    외부 장애로 기록되면 실패율 지표가 오염된다. 이유를 따로 남겨야 갈린다.
    """

    async def ok(*args: object, **kwargs: object) -> bool:
        return False

    monkeypatch.setattr(llm, "judge_answer", ok)

    with structlog.testing.capture_logs() as captured:
        await judge_service.judge(_question(), ".")

    assert _events(captured, "judge.fallback")[0]["reason"] == "empty_after_normalization"


@pytest.mark.asyncio
async def test_판정_한_건이_한_줄로_남는다(monkeypatch: pytest.MonkeyPatch) -> None:
    """**프롬프트 버전이 들어가는 것이 핵심이다.**

    프롬프트는 코드와 다른 주기로 바뀐다. 커밋 해시만으로는 어느 버전이 그 판정을
    냈는지 알 수 없어서, 프롬프트를 바꾼 효과를 사후에 볼 수 없다.
    """

    async def embed(text: str) -> list[float]:
        return [1.0, 0.0]

    async def high(vector: list[float], question_id: int) -> vector_store.AnchorMatch:
        return vector_store.AnchorMatch(similarity=0.99, rival_similarity=0.1)

    monkeypatch.setattr(embedding, "embed_one", embed)
    monkeypatch.setattr(vector_store, "match", high)

    with structlog.testing.capture_logs() as captured:
        await judge_service.judge(_question(), "서울")

    completed = _events(captured, "judge.completed")
    assert len(completed) == 1
    entry = completed[0]
    for field in (
        "method",
        "is_correct",
        "similarity",
        "embedding_model",
        "template_version",
        "judge_prompt",
        "judge_model",
        "duration_ms",
    ):
        assert field in entry, f"{field}가 판정 로그에 없다"
    assert entry["method"] == JudgeMethod.EMBEDDING
    assert entry["template_version"] == normalization.TEMPLATE_VERSION


@pytest.mark.asyncio
async def test_답변_원문은_로그에_안_남는다(monkeypatch: pytest.MonkeyPatch) -> None:
    """사용자 입력이고 로그는 평문으로 여러 곳에 복제된다."""

    async def embed(text: str) -> list[float]:
        return [1.0, 0.0]

    async def high(vector: list[float], question_id: int) -> vector_store.AnchorMatch:
        return vector_store.AnchorMatch(similarity=0.99, rival_similarity=0.1)

    monkeypatch.setattr(embedding, "embed_one", embed)
    monkeypatch.setattr(vector_store, "match", high)
    secret = "주민번호는 900101-1234567입니다"

    with structlog.testing.capture_logs() as captured:
        await judge_service.judge(_question(), secret)

    assert secret not in str(captured)
    assert "900101" not in str(captured)


def test_요청_ID가_모든_로그에_붙는다() -> None:
    """함수마다 ID를 넘겨줄 필요가 없다는 것이 contextvar를 쓰는 이유다."""
    log.configure(json_output=True)
    first = log.new_request_id()
    assert log.current_request_id() == first

    log.set_request_id("from-proxy")
    assert log.current_request_id() == "from-proxy"

    second = log.new_request_id()
    assert second != first
    assert len(second) == 12
