from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import models  # noqa: F401  create_all이 테이블을 알려면 모델이 임포트되어야 한다
from app.core.database import Base, engine
from app.core.exceptions import AppError
from app.routers import personas, quiz_sets, sessions


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # MVP 범위라 마이그레이션 도구를 두지 않았다.
    # 스키마 변경 이력을 남겨야 하는 시점(운영 데이터가 생기는 시점)에 Alembic으로 옮긴다
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(title="persona-quiz-agent", lifespan=lifespan)


@app.exception_handler(AppError)
async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    """서비스가 던진 도메인 에러를 공통 응답 래퍼로 바꾼다.

    이 핸들러가 있어서 서비스 코드가 HTTPException이나 상태 코드를 모른 채
    도메인 언어로만 실패를 표현할 수 있다.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"data": None, "error": {"code": exc.code, "message": exc.message}},
    )


app.include_router(personas.router)
app.include_router(quiz_sets.router)
app.include_router(sessions.router)
