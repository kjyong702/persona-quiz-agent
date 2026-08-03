import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import models  # noqa: F401  create_all이 테이블을 알려면 모델이 임포트되어야 한다
from app.core import credentials, log, prompts
from app.core.config import settings
from app.core.database import Base, engine
from app.core.exceptions import AppError
from app.routers import metrics, personas, quiz_sets, sessions


log.configure(json_output=settings.log_json, level=settings.log_level)
_log = log.get(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # MVP 범위라 마이그레이션 도구를 두지 않았다.
    # 스키마 변경 이력을 남겨야 하는 시점(운영 데이터가 생기는 시점)에 Alembic으로 옮긴다
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # 부팅 시 무엇을 들고 시작하는지 한 줄로 남긴다. 배포된 인스턴스가
    # 어느 모델과 어느 프롬프트로 도는지는 로그 첫 줄에서 확인할 수 있어야 한다
    _log.info(
        "app.started",
        embedding_model=settings.embedding_model,
        judge_model=settings.judge_model,
        judge_prompt=prompts.JUDGE_PROMPT,
        upper_threshold=settings.upper_threshold,
        lower_threshold=settings.lower_threshold,
        # 키 자체는 절대 안 남긴다. 지문이면 회전 여부를 구분하기에 충분하다
        api_key_fingerprint=credentials.fingerprint(credentials.current_api_key()),
    )
    yield


app = FastAPI(title="persona-quiz-agent", lifespan=lifespan)


@app.middleware("http")
async def attach_request_id(request: Request, call_next: Any) -> Any:
    """요청마다 ID를 심는다. 이후 이 요청에서 나오는 모든 로그에 자동으로 붙는다.

    답변 하나를 판정하는 데 임베딩 1회와 LLM 0~1회가 나간다. 로그가 흩어져 있으면
    **어느 판정에 속한 호출인지 알 수 없다.** contextvar에 심어두면 로거가
    알아서 붙이므로 함수마다 ID를 넘겨줄 필요가 없다.

    클라이언트가 `X-Request-ID`를 보내면 그걸 쓴다. 앞단에 프록시를 두면
    보통 거기서 붙여 보내고, 그래야 프록시 로그와 우리 로그가 이어진다.
    """
    incoming = request.headers.get("x-request-id")
    request_id = incoming[:64] if incoming else log.new_request_id()
    if incoming:
        log.set_request_id(request_id)

    started = time.perf_counter()
    response = await call_next(request)
    _log.info(
        "http.request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round((time.perf_counter() - started) * 1000, 1),
    )
    response.headers["X-Request-ID"] = request_id
    return response


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


app.include_router(metrics.router)
app.include_router(personas.router)
app.include_router(quiz_sets.router)
app.include_router(sessions.router)
