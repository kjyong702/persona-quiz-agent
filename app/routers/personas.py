from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.common import ApiResponse, ok
from app.schemas.persona import PersonaSummary
from app.services import persona_service

router = APIRouter(tags=["personas"])

DbSession = Annotated[AsyncSession, Depends(get_db)]


@router.get("/personas", response_model=ApiResponse[list[PersonaSummary]])
async def list_personas(db: DbSession) -> ApiResponse[list[PersonaSummary]]:
    return ok(await persona_service.list_personas(db))
