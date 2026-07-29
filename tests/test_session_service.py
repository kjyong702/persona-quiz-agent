"""세션 진행 서비스 테스트.

Phase 2의 핵심 로직은 next의 진행 규칙이다. 특히 "미답변 문제는 다시 준다"는
멱등 규칙이 깨지면 클라이언트가 next를 재시도하는 것만으로 문제를 건너뛴다.
"""

from collections.abc import Awaitable, Callable
from types import SimpleNamespace

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AppError, ErrorCode
from app.models import SessionStatus
from app.schemas.session import SessionCreateRequest
from app.services import session_service


async def _start(db: AsyncSession, seeded: SimpleNamespace) -> int:
    created = await session_service.start_session(
        db,
        SessionCreateRequest(quiz_set_id=seeded.quiz_set_id, persona_id=seeded.persona_id),
    )
    return created.session_id


# --- 세션 시작 ---


async def test_start_session_creates_session(
    db: AsyncSession, seeded: SimpleNamespace
) -> None:
    created = await session_service.start_session(
        db,
        SessionCreateRequest(quiz_set_id=seeded.quiz_set_id, persona_id=seeded.persona_id),
    )

    assert created.session_id > 0
    # 오프닝 멘트는 Phase 4에서 채운다
    assert created.host_message is None

    state = await session_service.get_state(db, created.session_id)
    assert state.status == SessionStatus.IN_PROGRESS
    assert state.current_order == 0
    assert state.total_questions == 3
    assert state.correct_count == 0


async def test_start_session_with_unknown_quiz_set(
    db: AsyncSession, seeded: SimpleNamespace
) -> None:
    with pytest.raises(AppError) as exc_info:
        await session_service.start_session(
            db, SessionCreateRequest(quiz_set_id=9999, persona_id=seeded.persona_id)
        )

    assert exc_info.value.code == ErrorCode.QUIZ_SET_NOT_FOUND
    assert exc_info.value.status_code == 404


async def test_start_session_with_unknown_persona(
    db: AsyncSession, seeded: SimpleNamespace
) -> None:
    with pytest.raises(AppError) as exc_info:
        await session_service.start_session(
            db, SessionCreateRequest(quiz_set_id=seeded.quiz_set_id, persona_id=9999)
        )

    assert exc_info.value.code == ErrorCode.PERSONA_NOT_FOUND
    assert exc_info.value.status_code == 404


async def test_start_session_with_empty_quiz_set(
    db: AsyncSession, seeded: SimpleNamespace
) -> None:
    """문제 없는 세트는 시작 시점에 막는다. 첫 next에서 곧바로 끝나면 원인을 알기 어렵다."""
    with pytest.raises(AppError) as exc_info:
        await session_service.start_session(
            db,
            SessionCreateRequest(
                quiz_set_id=seeded.empty_quiz_set_id, persona_id=seeded.persona_id
            ),
        )

    assert exc_info.value.code == ErrorCode.QUIZ_SET_EMPTY
    assert exc_info.value.status_code == 400


# --- 문제 출제 ---


async def test_next_serves_first_question(
    db: AsyncSession, seeded: SimpleNamespace
) -> None:
    session_id = await _start(db, seeded)

    question = await session_service.next_question(db, session_id)

    assert question.finished is False
    assert question.order_no == 1
    assert question.question_text == "1번 문제"
    assert (await session_service.get_state(db, session_id)).current_order == 1


async def test_next_repeats_unanswered_question(
    db: AsyncSession, seeded: SimpleNamespace
) -> None:
    """미답변 상태에서 next를 다시 부르면 같은 문제가 나오고 순서는 그대로다."""
    session_id = await _start(db, seeded)

    first = await session_service.next_question(db, session_id)
    again = await session_service.next_question(db, session_id)

    assert again.question_id == first.question_id
    assert again.order_no == 1
    assert (await session_service.get_state(db, session_id)).current_order == 1


async def test_next_advances_after_answer(
    db: AsyncSession,
    seeded: SimpleNamespace,
    record_answer: Callable[..., Awaitable[None]],
) -> None:
    session_id = await _start(db, seeded)
    first = await session_service.next_question(db, session_id)
    await record_answer(session_id, first.question_id)

    second = await session_service.next_question(db, session_id)

    assert second.order_no == 2
    assert second.question_id != first.question_id


async def test_next_finishes_after_last_question(
    db: AsyncSession,
    seeded: SimpleNamespace,
    record_answer: Callable[..., Awaitable[None]],
) -> None:
    session_id = await _start(db, seeded)
    for _ in range(3):
        question = await session_service.next_question(db, session_id)
        await record_answer(session_id, question.question_id)

    finished = await session_service.next_question(db, session_id)

    assert finished.finished is True
    assert finished.question_id is None
    assert finished.question_text is None
    assert (await session_service.get_state(db, session_id)).status == SessionStatus.FINISHED


async def test_next_on_finished_session_raises(
    db: AsyncSession,
    seeded: SimpleNamespace,
    record_answer: Callable[..., Awaitable[None]],
) -> None:
    session_id = await _start(db, seeded)
    for _ in range(3):
        question = await session_service.next_question(db, session_id)
        await record_answer(session_id, question.question_id)
    await session_service.next_question(db, session_id)  # 여기서 finished가 된다

    with pytest.raises(AppError) as exc_info:
        await session_service.next_question(db, session_id)

    assert exc_info.value.code == ErrorCode.SESSION_FINISHED
    assert exc_info.value.status_code == 400


async def test_next_on_unknown_session_raises(db: AsyncSession) -> None:
    with pytest.raises(AppError) as exc_info:
        await session_service.next_question(db, 9999)

    assert exc_info.value.code == ErrorCode.SESSION_NOT_FOUND


# --- 진행 상태 ---


async def test_state_counts_only_correct_answers(
    db: AsyncSession,
    seeded: SimpleNamespace,
    record_answer: Callable[..., Awaitable[None]],
) -> None:
    session_id = await _start(db, seeded)
    first = await session_service.next_question(db, session_id)
    await record_answer(session_id, first.question_id, is_correct=True)
    second = await session_service.next_question(db, session_id)
    await record_answer(session_id, second.question_id, is_correct=False)

    state = await session_service.get_state(db, session_id)

    assert state.correct_count == 1
    assert state.current_order == 2
    assert state.total_questions == 3


async def test_state_on_unknown_session_raises(db: AsyncSession) -> None:
    with pytest.raises(AppError) as exc_info:
        await session_service.get_state(db, 9999)

    assert exc_info.value.code == ErrorCode.SESSION_NOT_FOUND
    assert exc_info.value.status_code == 404
