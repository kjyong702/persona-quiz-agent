"""중복 제출과 DB 안전망.

**여기서 확인하는 것은 두 가지다.**

1. 같은 문항에 답변이 두 번 들어와도 데이터가 하나만 남는가
2. SQLite 외래키 검사가 실제로 켜져 있는가

둘 다 **선언만 해두고 안 지켜지기 쉬운 것들**이다. `UniqueConstraint`는 마이그레이션
없이 모델만 고치면 기존 DB에 안 걸리고, `ForeignKey`는 SQLite에서 PRAGMA를
안 켜면 문서일 뿐이다.
"""

import asyncio

import pytest
import pytest_asyncio
import structlog
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.database import Base, _enable_sqlite_foreign_keys
from app.models import JudgeMethod, Persona, Question, QuizSession, QuizSet
from app.repositories import session_repository
from app.schemas.judge import JudgeResult
from sqlalchemy import event


@pytest_asyncio.fixture
async def db_engine(tmp_path: object) -> AsyncEngine:
    """엔진을 따로 낸다.

    `session.get_bind()`는 **동기 Engine을 준다.** 비동기 세션이라도 바인딩된
    것은 sync_engine이라, 그걸로 새 세션을 만들면 AsyncEngine이 아니라고 거절당한다.
    동시 제출을 재현하려면 세션을 여러 개 만들어야 해서 엔진이 필요하다.
    """
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/t.db")
    # 운영과 같은 자리에 PRAGMA를 건다. 여기서 안 걸면 테스트만 통과하는
    # 상태가 되고, 정작 확인하려던 것을 확인하지 못한다
    event.listen(engine.sync_engine, "connect", _enable_sqlite_foreign_keys)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db(db_engine: AsyncEngine) -> AsyncSession:
    maker = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        session.add_all(
            [
                QuizSet(id=1, title="상식", description="테스트"),
                Persona(id=1, name="불꽃", personality="p", speech_style="s", reaction_style="r"),
            ]
        )
        await session.commit()
        session.add(Question(id=1, quiz_set_id=1, order_no=1, question_text="q", expected_answers=["a"]))
        session.add(QuizSession(id=1, quiz_set_id=1, persona_id=1))
        await session.commit()
        yield session


def _result(is_correct: bool = True) -> JudgeResult:
    return JudgeResult(
        is_correct=is_correct, judge_method=JudgeMethod.EMBEDDING, similarity=0.95
    )


@pytest.mark.asyncio
async def test_같은_문항에_두_번_넣으면_하나만_남는다(db: AsyncSession) -> None:
    first = await session_repository.create_answer(db, 1, 1, "서울", _result())
    second = await session_repository.create_answer(db, 1, 1, "서울", _result())

    # 두 번째는 새로 만들지 않고 기존 것을 돌려준다
    assert second.id == first.id

    count = await db.execute(text("SELECT COUNT(*) FROM session_answers"))
    assert count.scalar() == 1


@pytest.mark.asyncio
async def test_재전송에_에러_대신_같은_결과를_준다(db: AsyncSession) -> None:
    """네트워크가 끊겨 다시 보낸 사용자에게 409를 주는 것은 도움이 안 된다.

    이미 채점됐고 그 결과는 바뀌지 않으므로 같은 답을 주는 쪽이 맞다.
    """
    first = await session_repository.create_answer(db, 1, 1, "서울", _result(True))
    # 두 번째 요청의 판정 결과가 흔들려도(LLM은 비결정적이다) 처음 것이 유지된다
    second = await session_repository.create_answer(db, 1, 1, "서울", _result(False))

    assert second.is_correct == first.is_correct is True


@pytest.mark.asyncio
async def test_동시에_들어와도_하나만_남는다(
    db: AsyncSession, db_engine: AsyncEngine
) -> None:
    """**실제 경쟁을 만든다.**

    판정에 수 초가 걸리므로 창이 넓다. 순차 호출만 테스트하면 제약이 아니라
    조회 로직을 확인하는 셈이 된다.
    """
    maker = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def submit() -> int:
        async with maker() as own:
            answer = await session_repository.create_answer(own, 1, 1, "서울", _result())
            return answer.id

    ids = await asyncio.gather(*(submit() for _ in range(5)), return_exceptions=True)
    failures = [r for r in ids if isinstance(r, Exception)]
    assert not failures, f"동시 제출이 예외를 냈다: {failures}"
    assert len(set(ids)) == 1, f"행이 여러 개 생겼다: {ids}"

    count = await db.execute(text("SELECT COUNT(*) FROM session_answers"))
    assert count.scalar() == 1


@pytest.mark.asyncio
async def test_외래키가_실제로_켜져_있다(db: AsyncSession) -> None:
    """선언만 있고 PRAGMA를 안 켜면 `ForeignKey`는 문서일 뿐이다."""
    pragma = await db.execute(text("PRAGMA foreign_keys"))
    assert pragma.scalar() == 1


@pytest.mark.asyncio
async def test_없는_세션에_답변을_넣으면_거부된다(db: AsyncSession) -> None:
    """**음성 대조에 해당한다.**

    PRAGMA가 꺼져 있으면 이 삽입이 그냥 통과한다. 그리고 나중에 조회에서
    빈 결과가 나올 뿐이라 원인을 찾기 어렵다.
    """
    with pytest.raises(IntegrityError):
        await session_repository.create_answer(db, 999, 1, "서울", _result())


@pytest.mark.asyncio
async def test_버려지는_판정이_다르면_그_사실을_남긴다(db: AsyncSession) -> None:
    """**중복 제출은 공짜 재현 실험이다.**

    같은 답변을 두 번 판정한 것이므로 결과가 다르면 비결정성의 실제 관측이다.
    Phase 5에서 266건을 세 번씩 돌려 유료로 잰 것이 운영에서는 저절로 생긴다.

    결과는 바꾸지 않는다. 먼저 저장된 것이 진실이고, 로그에만 남긴다.
    """
    await session_repository.create_answer(db, 1, 1, "서울", _result(True))

    with structlog.testing.capture_logs() as captured:
        second = await session_repository.create_answer(db, 1, 1, "서울", _result(False))

    duplicate = [e for e in captured if e.get("event") == "answer.duplicate"]
    assert len(duplicate) == 1
    assert duplicate[0]["verdict_changed"] is True
    assert duplicate[0]["stored_is_correct"] is True
    assert duplicate[0]["discarded_is_correct"] is False
    # 판정은 바뀌지 않는다
    assert second.is_correct is True


@pytest.mark.asyncio
async def test_판정이_같으면_바뀌지_않았다고_남긴다(db: AsyncSession) -> None:
    """대부분은 이쪽이다. 그래야 `verdict_changed`가 참인 경우를 셀 수 있다."""
    await session_repository.create_answer(db, 1, 1, "서울", _result(True))

    with structlog.testing.capture_logs() as captured:
        await session_repository.create_answer(db, 1, 1, "서울", _result(True))

    duplicate = [e for e in captured if e.get("event") == "answer.duplicate"]
    assert duplicate[0]["verdict_changed"] is False
