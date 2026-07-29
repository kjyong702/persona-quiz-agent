from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories import quiz_repository
from app.schemas.quiz import QuizSetSummary


async def list_quiz_sets(db: AsyncSession) -> list[QuizSetSummary]:
    rows = await quiz_repository.list_sets_with_counts(db)
    return [
        QuizSetSummary(
            id=quiz_set.id,
            title=quiz_set.title,
            description=quiz_set.description,
            question_count=question_count,
        )
        for quiz_set, question_count in rows
    ]
