from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.common import ApiResponse, ok
from app.schemas.quiz import QuizSetSummary
from app.services import quiz_service

router = APIRouter(tags=["quiz-sets"])

DbSession = Annotated[AsyncSession, Depends(get_db)]


@router.get("/quiz-sets", response_model=ApiResponse[list[QuizSetSummary]])
async def list_quiz_sets(db: DbSession) -> ApiResponse[list[QuizSetSummary]]:
    return ok(await quiz_service.list_quiz_sets(db))
