from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import log
from app.models import QuizSession, SessionAnswer
from app.schemas.judge import JudgeResult


_log = log.get(__name__)


async def create(db: AsyncSession, quiz_set_id: int, persona_id: int) -> QuizSession:
    session = QuizSession(quiz_set_id=quiz_set_id, persona_id=persona_id)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def get(db: AsyncSession, session_id: int) -> QuizSession | None:
    return await db.get(QuizSession, session_id)


async def save(db: AsyncSession, session: QuizSession) -> None:
    db.add(session)
    await db.commit()


async def create_answer(
    db: AsyncSession,
    session_id: int,
    question_id: int,
    answer_text: str,
    result: JudgeResult,
) -> SessionAnswer:
    answer = SessionAnswer(
        session_id=session_id,
        question_id=question_id,
        answer_text=answer_text,
        is_correct=result.is_correct,
        judge_method=result.judge_method,
        similarity=result.similarity,
        rival_similarity=result.rival_similarity,
        embedding_model=result.embedding_model,
        template_version=result.template_version,
    )
    db.add(answer)
    try:
        await db.commit()
    except IntegrityError:
        # 같은 문항에 답변이 이미 있다. 같은 답이 동시에 두 번 들어왔거나
        # 클라이언트가 재전송한 것이다.
        #
        # **409로 거절하지 않고 기존 결과를 돌려준다.** 네트워크가 끊겨 다시 보낸
        # 사용자에게 에러를 주는 것은 도움이 안 된다. 이미 채점됐고 그 결과는
        # 바뀌지 않으므로 같은 답을 주는 쪽이 맞다.
        #
        # ⚠️ **제약은 마지막 방어선이지 첫 방어선이 아니다.** 두 요청이 동시에 오면
        # 둘 다 LLM을 부르고 둘 중 하나만 저장에 성공한다. 데이터는 하나지만
        # 비용은 두 번 났다. 실행 자체를 막으려면 판정 전에 자리를 선점해야 한다.
        # 상세는 docs/notes/duplicate-submit.md
        await db.rollback()
        _log.warning(
            "answer.duplicate",
            session_id=session_id,
            question_id=question_id,
        )
        existing = await get_answer(db, session_id, question_id)
        if existing is None:
            # 제약이 걸렸는데 조회가 비었다. 다른 제약이 깨진 것이므로 삼키지 않는다
            raise
        return existing
    await db.refresh(answer)
    return answer


async def count_correct(db: AsyncSession, session_id: int) -> int:
    stmt = select(func.count(SessionAnswer.id)).where(
        SessionAnswer.session_id == session_id,
        SessionAnswer.is_correct.is_(True),
    )
    return (await db.execute(stmt)).scalar_one()


async def get_answer(
    db: AsyncSession, session_id: int, question_id: int
) -> SessionAnswer | None:
    """해당 문제에 이미 답변했는지 확인용.

    next의 멱등 규칙(미답변이면 같은 문제를 다시 준다)이 이 조회에 걸려 있다.
    """
    stmt = select(SessionAnswer).where(
        SessionAnswer.session_id == session_id,
        SessionAnswer.question_id == question_id,
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def get_answer(
    db: AsyncSession, session_id: int, question_id: int
) -> SessionAnswer | None:
    """이미 채점된 답변. 중복 제출을 멱등으로 처리할 때 쓴다."""
    result = await db.execute(
        select(SessionAnswer).where(
            SessionAnswer.session_id == session_id,
            SessionAnswer.question_id == question_id,
        )
    )
    return result.scalar_one_or_none()
