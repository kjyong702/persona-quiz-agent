from collections.abc import Sequence

from sqlalchemy import Row, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Question, QuizSet


async def list_sets_with_counts(db: AsyncSession) -> Sequence[Row[tuple[QuizSet, int]]]:
    """세트 목록과 각 세트의 문제 수를 한 번에 가져온다.

    세트마다 count 쿼리를 따로 날리면 세트 수만큼 쿼리가 늘어난다.
    문제가 없는 세트도 목록에 나와야 하므로 outer join이다.
    """
    stmt = (
        select(QuizSet, func.count(Question.id).label("question_count"))
        .outerjoin(Question, Question.quiz_set_id == QuizSet.id)
        .group_by(QuizSet.id)
        .order_by(QuizSet.id)
    )
    result = await db.execute(stmt)
    return result.all()


async def get_set(db: AsyncSession, quiz_set_id: int) -> QuizSet | None:
    return await db.get(QuizSet, quiz_set_id)


async def count_questions(db: AsyncSession, quiz_set_id: int) -> int:
    stmt = select(func.count(Question.id)).where(Question.quiz_set_id == quiz_set_id)
    return (await db.execute(stmt)).scalar_one()


async def get_question_by_order(
    db: AsyncSession, quiz_set_id: int, order_no: int
) -> Question | None:
    stmt = select(Question).where(
        Question.quiz_set_id == quiz_set_id,
        Question.order_no == order_no,
    )
    return (await db.execute(stmt)).scalar_one_or_none()
