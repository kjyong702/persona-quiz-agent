"""페르소나와 퀴즈 세트 조회 서비스 테스트."""

from types import SimpleNamespace

from sqlalchemy.ext.asyncio import AsyncSession

from app.services import persona_service, quiz_service


async def test_list_personas_hides_reaction_style(
    db: AsyncSession, seeded: SimpleNamespace
) -> None:
    personas = await persona_service.list_personas(db)

    assert len(personas) == 1
    assert personas[0].name == "불꽃"
    # reaction_style은 프롬프트 조립용 내부 값이라 응답 스키마에 없다
    assert not hasattr(personas[0], "reaction_style")


async def test_list_quiz_sets_counts_questions(
    db: AsyncSession, seeded: SimpleNamespace
) -> None:
    quiz_sets = await quiz_service.list_quiz_sets(db)
    counts = {qs.id: qs.question_count for qs in quiz_sets}

    assert counts[seeded.quiz_set_id] == 3


async def test_list_quiz_sets_includes_empty_set(
    db: AsyncSession, seeded: SimpleNamespace
) -> None:
    """문제가 없는 세트도 목록에서 빠지지 않는다 (inner join이면 사라진다)."""
    quiz_sets = await quiz_service.list_quiz_sets(db)
    counts = {qs.id: qs.question_count for qs in quiz_sets}

    assert seeded.empty_quiz_set_id in counts
    assert counts[seeded.empty_quiz_set_id] == 0
