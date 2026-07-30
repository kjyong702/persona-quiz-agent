"""시드 무결성 검사 테스트.

시드의 실패는 예외로 터지는 실패가 아니라 **성공처럼 보이는 실패**가 위험하다.
문항은 들어갔는데 앵커가 없으면 판정이 전부 LLM 폴백으로 가면서 비용은 오르고
하이브리드는 꺼지는데, 어디에도 신호가 없다. 그 상태를 만들 수 있는 경로마다
검사가 걸려 있는지를 여기서 본다.
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from app.core import embedding, vector_store
from app.models import Question
from app.repositories import quiz_repository
from scripts import seed


def _question(qid: int, answers: list[str]) -> Question:
    return Question(
        id=qid,
        quiz_set_id=1,
        order_no=qid,
        question_text=f"문항 {qid}",
        expected_answers=answers,
    )


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """DB와 외부 호출을 전부 갈아끼운다. 여기서 보는 것은 검사 로직뿐이다."""
    state = SimpleNamespace(questions=[], stored=None, replaced=None)

    @asynccontextmanager
    async def fake_session():  # noqa: ANN202
        yield None

    async def fake_list_all_questions(db):  # noqa: ANN001, ARG001
        return state.questions

    async def fake_embed(texts: list[str]) -> list[list[float]]:
        return [[float(i)] for i in range(len(texts))]

    async def fake_replace_anchors(anchors) -> None:  # noqa: ANN001
        state.replaced = anchors

    async def fake_count() -> int:
        # 지정이 없으면 실제로 넣은 수를 그대로 돌려준다 (정상 경로)
        return state.stored if state.stored is not None else len(state.replaced or [])

    monkeypatch.setattr(seed, "SessionLocal", fake_session)
    monkeypatch.setattr(quiz_repository, "list_all_questions", fake_list_all_questions)
    monkeypatch.setattr(embedding, "embed", fake_embed)
    monkeypatch.setattr(vector_store, "replace_anchors", fake_replace_anchors)
    monkeypatch.setattr(vector_store, "count", fake_count)
    return state


@pytest.mark.asyncio
async def test_정상_적재(stub: SimpleNamespace) -> None:
    stub.questions = [_question(1, ["서울", "서울특별시"]), _question(2, ["목성"])]

    assert await seed._load_anchors() == 3
    assert [a.id for a in stub.replaced] == ["q1-a0", "q1-a1", "q2-a0"]


@pytest.mark.asyncio
async def test_문항이_없으면_예외(stub: SimpleNamespace) -> None:
    """DB 시드가 실패했는데 앵커 단계로 넘어온 경우다."""
    stub.questions = []

    with pytest.raises(seed.SeedIntegrityError, match="문항이 하나도 없습니다"):
        await seed._load_anchors()


@pytest.mark.asyncio
async def test_기대_정답이_빈_문항이_있으면_예외(stub: SimpleNamespace) -> None:
    """이 문항만 앵커가 안 생겨 조용히 LLM 폴백으로 빠진다. 그래서 끊는다."""
    stub.questions = [_question(1, ["서울"]), _question(2, [])]

    with pytest.raises(seed.SeedIntegrityError, match=r"\[2\]"):
        await seed._load_anchors()


@pytest.mark.asyncio
async def test_적재_수가_스토어와_다르면_예외(stub: SimpleNamespace) -> None:
    """넣었다고 믿지 않고 스토어에 물어본 값과 대조한다."""
    stub.questions = [_question(1, ["서울", "서울특별시"])]
    stub.stored = 1  # 2건 넣었는데 1건만 남은 상황

    with pytest.raises(seed.SeedIntegrityError, match="맞지 않습니다"):
        await seed._load_anchors()
