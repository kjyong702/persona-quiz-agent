"""답변 제출 흐름 테스트.

판정 자체는 test_judge_service.py가 본다. 여기서 보는 것은
세션 규칙(언제 답할 수 있는가)과 판정 근거가 DB에 남는가다.
"""

from types import SimpleNamespace

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import embedding, llm, normalization, vector_store
from app.core.config import settings
from app.core.exceptions import AppError, ErrorCode
from app.models import JudgeMethod, SessionAnswer
from app.schemas.session import AnswerRequest, SessionCreateRequest
from app.services import session_service


@pytest.fixture
def stub_embedding_correct(monkeypatch: pytest.MonkeyPatch) -> None:
    """임베딩만으로 정답 확정되는 상황. LLM은 아예 부르지 못하게 막아둔다."""

    async def fake_embed_one(text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def fake_match(vector: list[float], question_id: int) -> vector_store.AnchorMatch:
        return vector_store.AnchorMatch(
            similarity=settings.upper_threshold + 0.05, rival_similarity=0.1
        )

    async def forbidden(*args: object, **kwargs: object) -> bool:
        raise AssertionError("이 경로에서는 LLM을 부르면 안 된다")

    monkeypatch.setattr(embedding, "embed_one", fake_embed_one)
    monkeypatch.setattr(vector_store, "match", fake_match)
    monkeypatch.setattr(llm, "judge_answer", forbidden)


async def _start_and_serve(db: AsyncSession, seeded: SimpleNamespace) -> int:
    created = await session_service.start_session(
        db,
        SessionCreateRequest(quiz_set_id=seeded.quiz_set_id, persona_id=seeded.persona_id),
    )
    await session_service.next_question(db, created.session_id)
    return created.session_id


async def test_answer_records_judgement_evidence(
    db: AsyncSession, seeded: SimpleNamespace, stub_embedding_correct: None
) -> None:
    session_id = await _start_and_serve(db, seeded)

    result = await session_service.submit_answer(
        db, session_id, AnswerRequest(answer="정답1")
    )

    assert result.is_correct is True
    assert result.judge_method == JudgeMethod.EMBEDDING
    assert result.host_message is None  # Phase 4에서 채운다

    stored = (await db.execute(select(SessionAnswer))).scalars().all()
    assert len(stored) == 1
    assert stored[0].answer_text == "정답1"
    assert stored[0].judge_method == JudgeMethod.EMBEDDING
    assert stored[0].similarity == pytest.approx(settings.upper_threshold + 0.05)
    assert stored[0].rival_similarity == pytest.approx(0.1)
    assert stored[0].embedding_model == settings.embedding_model
    assert stored[0].template_version == normalization.TEMPLATE_VERSION


async def test_answer_updates_correct_count(
    db: AsyncSession, seeded: SimpleNamespace, stub_embedding_correct: None
) -> None:
    session_id = await _start_and_serve(db, seeded)
    await session_service.submit_answer(db, session_id, AnswerRequest(answer="정답1"))

    state = await session_service.get_state(db, session_id)

    assert state.correct_count == 1


async def test_answer_then_next_advances(
    db: AsyncSession, seeded: SimpleNamespace, stub_embedding_correct: None
) -> None:
    """답변하면 next의 멱등 규칙이 풀려 다음 문제로 넘어간다."""
    session_id = await _start_and_serve(db, seeded)
    await session_service.submit_answer(db, session_id, AnswerRequest(answer="정답1"))

    question = await session_service.next_question(db, session_id)

    assert question.order_no == 2


async def test_answer_without_question_raises(
    db: AsyncSession, seeded: SimpleNamespace, stub_embedding_correct: None
) -> None:
    created = await session_service.start_session(
        db,
        SessionCreateRequest(quiz_set_id=seeded.quiz_set_id, persona_id=seeded.persona_id),
    )

    with pytest.raises(AppError) as exc_info:
        await session_service.submit_answer(
            db, created.session_id, AnswerRequest(answer="서울")
        )

    assert exc_info.value.code == ErrorCode.NO_ACTIVE_QUESTION


async def test_answering_twice_raises(
    db: AsyncSession, seeded: SimpleNamespace, stub_embedding_correct: None
) -> None:
    """판정을 덮어쓰면 평가 데이터가 오염된다."""
    session_id = await _start_and_serve(db, seeded)
    await session_service.submit_answer(db, session_id, AnswerRequest(answer="정답1"))

    with pytest.raises(AppError) as exc_info:
        await session_service.submit_answer(
            db, session_id, AnswerRequest(answer="다시 답변")
        )

    assert exc_info.value.code == ErrorCode.NO_ACTIVE_QUESTION


async def test_answer_on_finished_session_raises(
    db: AsyncSession, seeded: SimpleNamespace, stub_embedding_correct: None
) -> None:
    session_id = await _start_and_serve(db, seeded)
    for _ in range(3):
        await session_service.submit_answer(db, session_id, AnswerRequest(answer="정답"))
        await session_service.next_question(db, session_id)

    with pytest.raises(AppError) as exc_info:
        await session_service.submit_answer(db, session_id, AnswerRequest(answer="정답"))

    assert exc_info.value.code == ErrorCode.SESSION_FINISHED


async def test_answer_endpoint_shape(
    client: AsyncClient, seeded: SimpleNamespace, stub_embedding_correct: None
) -> None:
    """라우터를 거친 응답에 내부 값이 새지 않는지 본다."""
    created = await client.post(
        "/sessions",
        json={"quiz_set_id": seeded.quiz_set_id, "persona_id": seeded.persona_id},
    )
    session_id = created.json()["data"]["session_id"]
    await client.post(f"/sessions/{session_id}/next")

    response = await client.post(
        f"/sessions/{session_id}/answer", json={"answer": "정답1"}
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["is_correct"] is True
    assert data["judge_method"] == JudgeMethod.EMBEDDING
    # 재현용 값은 DB에만 남고 응답에는 없다
    assert "embedding_model" not in data
    assert "template_version" not in data
    assert "rival_similarity" not in data
