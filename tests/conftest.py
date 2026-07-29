from collections.abc import AsyncGenerator, Awaitable, Callable
from types import SimpleNamespace

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.database import Base, get_db
from app.main import app
from app.models import JudgeMethod, Persona, Question, QuizSet, SessionAnswer


@pytest_asyncio.fixture
async def db() -> AsyncGenerator[AsyncSession, None]:
    """테스트마다 비어 있는 인메모리 DB를 새로 만든다.

    StaticPool을 쓰는 이유: 인메모리 SQLite는 연결마다 별개의 DB가 만들어진다.
    풀을 고정하지 않으면 테이블을 만든 연결과 조회하는 연결이 달라져서
    "no such table"이 난다.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session

    await engine.dispose()


@pytest_asyncio.fixture
async def seeded(db: AsyncSession) -> SimpleNamespace:
    """테스트용 최소 데이터. 문제 3개짜리 세트와 빈 세트를 함께 둔다."""
    persona = Persona(
        name="불꽃",
        personality="열혈",
        speech_style="반말",
        reaction_style="크게 환호",
    )
    quiz_set = QuizSet(title="테스트 세트", description="문제 3개")
    empty_set = QuizSet(title="빈 세트", description="문제 없음")
    db.add_all([persona, quiz_set, empty_set])
    await db.flush()

    questions = [
        Question(
            quiz_set_id=quiz_set.id,
            order_no=order_no,
            question_text=f"{order_no}번 문제",
            expected_answers=[f"정답{order_no}"],
        )
        for order_no in (1, 2, 3)
    ]
    db.add_all(questions)
    await db.commit()

    return SimpleNamespace(
        persona_id=persona.id,
        quiz_set_id=quiz_set.id,
        empty_quiz_set_id=empty_set.id,
        question_ids=[q.id for q in questions],
    )


@pytest_asyncio.fixture
async def client(db: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """라우터 테스트용 클라이언트. 앱의 DB 의존성을 테스트 세션으로 갈아끼운다."""

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client
    app.dependency_overrides.clear()


@pytest.fixture
def record_answer(db: AsyncSession) -> Callable[..., Awaitable[None]]:
    """답변 기록 헬퍼.

    답변 저장은 Phase 3(판정 파이프라인)에서 서비스로 들어온다.
    Phase 2에서는 "이미 답변한 상태"를 만들기 위해 테스트가 직접 넣는다.
    """

    async def _record(
        session_id: int, question_id: int, *, is_correct: bool = True
    ) -> None:
        db.add(
            SessionAnswer(
                session_id=session_id,
                question_id=question_id,
                answer_text="테스트 답변",
                is_correct=is_correct,
                judge_method=JudgeMethod.EMBEDDING,
                similarity=0.9,
            )
        )
        await db.commit()

    return _record
