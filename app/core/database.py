from collections.abc import AsyncGenerator
from datetime import UTC, datetime

from typing import Any

from sqlalchemy import DateTime, event
from sqlalchemy.types import TypeDecorator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.config import settings


class Base(DeclarativeBase):
    pass


class UtcDateTime(TypeDecorator):
    """UTC로 저장하고 UTC로 돌려주는 datetime.

    **`DateTime(timezone=True)`만으로는 SQLite에서 지켜지지 않는다.**
    SQLite에 datetime 타입이 없어서 문자열로 저장되는데, 그 문자열에
    시간대 표시가 안 들어간다. 그래서 다시 읽으면 tzinfo가 사라진다.

        넣을 때     2026-08-03 17:46:06 tzinfo=UTC
        DB 저장값   '2026-08-03 17:46:06.999397'    <- 시간대 표시 없음
        다시 읽으면 2026-08-03 17:46:06 tzinfo=None <- 사라졌다

    **조용히 틀리지 않고 시끄럽게 터지는 종류라 그나마 낫다.**
    naive와 aware를 비교하면 파이썬이 TypeError를 낸다. 그래도 세션 만료 시각을
    계산하거나 응답으로 직렬화할 때 터지므로 미리 막는다.

    Postgres로 옮기면 `TIMESTAMPTZ`가 있어서 이 래퍼가 필요 없어진다.
    다만 있어도 해가 없으므로 그대로 두면 된다.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: object) -> datetime | None:
        """저장 직전. tzinfo가 없으면 UTC로 간주하고, 있으면 UTC로 변환한다."""
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: object) -> datetime | None:
        """읽은 직후. 시간대가 없으면 UTC를 붙인다.

        저장할 때 UTC로 바꿔뒀으므로 이 값은 UTC가 맞다.
        **DB가 시간대를 안 알려주니 우리가 아는 사실을 되돌려주는 것이다.**
        """
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime,
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


@event.listens_for(engine.sync_engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection: Any, _record: Any) -> None:
    """SQLite는 외래키 검사가 **기본으로 꺼져 있다.**

    선언만 해두고 켜지 않으면 `ForeignKey`가 문서일 뿐 아무것도 안 막는다.
    없는 세션 ID로 답변을 넣어도 그냥 들어가고, 그때는 조회에서 빈 결과가 나올
    뿐이라 원인을 찾기 어렵다.

    **연결마다 걸어야 한다.** 커넥션 풀이 새 연결을 만들 때마다 초기화되므로
    한 번 실행하는 것으로는 안 된다. 그래서 엔진의 connect 이벤트에 붙인다.

    비동기 엔진에는 이벤트를 직접 못 걸어서 `sync_engine`을 쓴다. aiosqlite가
    안에서 동기 드라이버를 돌리는 구조라 실제 연결은 그쪽에서 만들어진다.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as db:
        yield db
