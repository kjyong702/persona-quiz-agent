from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import DomainError, ErrorCode, NotFoundError
from app.models import Question, QuizSession, SessionStatus
from app.repositories import persona_repository, quiz_repository, session_repository
from app.schemas.session import (
    AnswerRequest,
    AnswerResult,
    NextQuestion,
    SessionCreated,
    SessionCreateRequest,
    SessionState,
)
from app.services import judge_service


async def start_session(db: AsyncSession, request: SessionCreateRequest) -> SessionCreated:
    quiz_set = await quiz_repository.get_set(db, request.quiz_set_id)
    if quiz_set is None:
        raise NotFoundError(
            ErrorCode.QUIZ_SET_NOT_FOUND,
            f"퀴즈 세트 {request.quiz_set_id}를 찾을 수 없습니다",
        )

    persona = await persona_repository.get(db, request.persona_id)
    if persona is None:
        raise NotFoundError(
            ErrorCode.PERSONA_NOT_FOUND,
            f"페르소나 {request.persona_id}를 찾을 수 없습니다",
        )

    # 문제가 없는 세트로 세션을 열면 첫 next에서 곧바로 종료된다.
    # 시작 시점에 막는 편이 클라이언트가 원인을 알기 쉽다
    if await quiz_repository.count_questions(db, quiz_set.id) == 0:
        raise DomainError(
            ErrorCode.QUIZ_SET_EMPTY,
            f"퀴즈 세트 {quiz_set.id}에 문제가 없습니다",
        )

    session = await session_repository.create(db, quiz_set.id, persona.id)
    # 오프닝 멘트는 Phase 4 페르소나 레이어에서 채운다
    return SessionCreated(session_id=session.id)


async def get_state(db: AsyncSession, session_id: int) -> SessionState:
    session = await _get_session_or_raise(db, session_id)
    return SessionState(
        status=session.status,
        current_order=session.current_order,
        total_questions=await quiz_repository.count_questions(db, session.quiz_set_id),
        correct_count=await session_repository.count_correct(db, session.id),
    )


async def next_question(db: AsyncSession, session_id: int) -> NextQuestion:
    session = await _get_session_or_raise(db, session_id)

    if session.status == SessionStatus.FINISHED:
        raise DomainError(
            ErrorCode.SESSION_FINISHED,
            f"세션 {session_id}는 이미 종료되었습니다",
        )

    pending = await _pending_question(db, session)
    if pending is not None:
        # 미답변 문제가 남아 있으면 같은 문제를 다시 준다 (api-spec 멱등 규칙).
        # current_order를 올리지 않으므로 next를 연타해도 문제를 건너뛰지 않는다
        return _to_response(pending)

    next_order = session.current_order + 1
    question = await quiz_repository.get_question_by_order(db, session.quiz_set_id, next_order)

    if question is None:
        session.status = SessionStatus.FINISHED
        await session_repository.save(db, session)
        # 마무리 멘트는 Phase 4에서 채운다
        return NextQuestion(finished=True)

    session.current_order = next_order
    await session_repository.save(db, session)
    return _to_response(question)


async def submit_answer(
    db: AsyncSession, session_id: int, request: AnswerRequest
) -> AnswerResult:
    session = await _get_session_or_raise(db, session_id)

    if session.status == SessionStatus.FINISHED:
        raise DomainError(
            ErrorCode.SESSION_FINISHED,
            f"세션 {session_id}는 이미 종료되었습니다",
        )

    question = await _active_question_or_raise(db, session)
    result = await judge_service.judge(question, request.answer)
    await session_repository.create_answer(
        db, session.id, question.id, request.answer, result
    )

    return AnswerResult(
        is_correct=result.is_correct,
        judge_method=result.judge_method,
        similarity=result.similarity,
        # 리액션 멘트는 Phase 4에서 채운다
        host_message=None,
    )


async def _active_question_or_raise(db: AsyncSession, session: QuizSession) -> Question:
    """지금 답변할 수 있는 문제. 없으면 NO_ACTIVE_QUESTION."""
    if session.current_order == 0:
        raise DomainError(
            ErrorCode.NO_ACTIVE_QUESTION,
            "아직 출제된 문제가 없습니다. 먼저 다음 문제를 받으세요",
        )

    question = await quiz_repository.get_question_by_order(
        db, session.quiz_set_id, session.current_order
    )
    if question is None:
        raise DomainError(
            ErrorCode.NO_ACTIVE_QUESTION,
            "출제된 문제를 찾을 수 없습니다",
        )

    if await session_repository.get_answer(db, session.id, question.id) is not None:
        # 같은 문제에 두 번 답할 수 없다. 판정을 덮어쓰면 평가 데이터가 오염된다
        raise DomainError(
            ErrorCode.NO_ACTIVE_QUESTION,
            "이미 답변한 문제입니다. 다음 문제를 받으세요",
        )

    return question


async def _get_session_or_raise(db: AsyncSession, session_id: int) -> QuizSession:
    session = await session_repository.get(db, session_id)
    if session is None:
        raise NotFoundError(
            ErrorCode.SESSION_NOT_FOUND,
            f"세션 {session_id}를 찾을 수 없습니다",
        )
    return session


async def _pending_question(db: AsyncSession, session: QuizSession) -> Question | None:
    """이미 출제했지만 아직 답변하지 않은 문제. 없으면 None."""
    if session.current_order == 0:
        return None

    current = await quiz_repository.get_question_by_order(
        db, session.quiz_set_id, session.current_order
    )
    if current is None:
        return None

    answer = await session_repository.get_answer(db, session.id, current.id)
    return current if answer is None else None


def _to_response(question: Question) -> NextQuestion:
    return NextQuestion(
        finished=False,
        question_id=question.id,
        order_no=question.order_no,
        question_text=question.question_text,
        # 출제 멘트는 Phase 4에서 채운다
        host_message=None,
    )
