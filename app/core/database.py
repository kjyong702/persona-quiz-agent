from collections.abc import AsyncGenerator
from datetime import UTC, datetime

from sqlalchemy import DateTime
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.config import settings


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )


# 동기 드라이버 대신 aiosqlite를 쓰는 이유: Phase 3.5에서 LLM 호출 흐름을
# 세마포어와 레이트 리미터로 제어한다. 요청 경로 중간에 동기 DB 호출이 섞이면
# 이벤트 루프가 막혀 동시성 제어 자체가 의미를 잃는다.
# **풀 크기가 곧 동시 외부 호출 상한이다.** 요청 스코프 세션이 LLM 응답을
# 기다리는 내내 커넥션을 점유하므로, 풀이 작으면 게이트를 아무리 열어도
# 그만큼밖에 못 나간다. 기본값(5+10)으로 실측했을 때 처리량이 11.1건/초에
# 묶였고 그 값은 풀 15를 LLM 지연으로 나눈 것과 일치했다.
#
# 근본 해법은 외부 호출 전에 커넥션을 놓는 것이지만, 그러면 트랜잭션 경계를
# 다시 설계해야 한다. 지금은 풀을 설정으로 빼서 병목 위치를 드러내는 데까지 한다
engine = create_async_engine(
    settings.database_url,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
)
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as db:
        yield db
