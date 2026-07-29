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
engine = create_async_engine(settings.database_url)
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as db:
        yield db
