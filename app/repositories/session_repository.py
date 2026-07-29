from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import QuizSession, SessionAnswer


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
