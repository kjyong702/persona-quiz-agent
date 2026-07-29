from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.common import ApiResponse, ok
from app.schemas.session import (
    NextQuestion,
    SessionCreated,
    SessionCreateRequest,
    SessionState,
)
from app.services import session_service

router = APIRouter(prefix="/sessions", tags=["sessions"])

DbSession = Annotated[AsyncSession, Depends(get_db)]


@router.post(
    "",
    response_model=ApiResponse[SessionCreated],
    status_code=status.HTTP_201_CREATED,
)
async def start_session(
    request: SessionCreateRequest, db: DbSession
) -> ApiResponse[SessionCreated]:
    return ok(await session_service.start_session(db, request))


@router.get("/{session_id}", response_model=ApiResponse[SessionState])
async def get_session(session_id: int, db: DbSession) -> ApiResponse[SessionState]:
    return ok(await session_service.get_state(db, session_id))


@router.post("/{session_id}/next", response_model=ApiResponse[NextQuestion])
async def next_question(session_id: int, db: DbSession) -> ApiResponse[NextQuestion]:
    return ok(await session_service.next_question(db, session_id))
