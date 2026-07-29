"""벡터 스토어 통합 테스트. Chroma를 실제로 띄운다.

임베딩만 손으로 만든 벡터로 대체하고 스토어는 진짜를 쓴다.
여기서 확인하는 것은 코사인 거리를 유사도로 뒤집는 환산과 question_id 필터다.

이 두 가지는 목으로는 절대 안 잡힌다. 거리와 유사도를 뒤집어 쓰면
모든 판정이 반대로 나오는데 단위 테스트는 전부 통과한다.
"""

from types import ModuleType

import pytest

from app.core import embedding, llm, vector_store
from app.core.config import settings
from app.models import JudgeMethod, Question
from app.services import judge_service

# 서로 직교하는 3차원 벡터. 코사인 유사도가 손으로 계산된다
Q1_VECTOR = [1.0, 0.0, 0.0]
Q2_VECTOR = [0.0, 1.0, 0.0]
DIAGONAL = [0.7071067811865476, 0.7071067811865476, 0.0]  # 둘 사이 45도


@pytest.fixture
def store(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setattr(settings, "chroma_path", str(tmp_path / "chroma"))  # type: ignore[operator]
    monkeypatch.setattr(vector_store, "_client", None)
    return vector_store


def _anchor(anchor_id: str, question_id: int, vector: list[float]) -> vector_store.Anchor:
    return vector_store.Anchor(
        id=anchor_id,
        embedding=vector,
        document=f"문서 {anchor_id}",
        metadata={"question_id": question_id, "quiz_set_id": 1},
    )


def _question() -> Question:
    return Question(
        id=1,
        quiz_set_id=1,
        order_no=1,
        question_text="대한민국의 수도는 어디인가요?",
        expected_answers=["서울", "서울특별시"],
    )


async def test_identical_vector_scores_one(store: ModuleType) -> None:
    """같은 방향 벡터의 코사인 유사도는 1이다.

    Chroma가 주는 것은 거리(1 - 유사도)라 그대로 쓰면 정답이 0점이 된다.
    """
    await store.replace_anchors([_anchor("q1-a0", 1, Q1_VECTOR)])

    match = await store.match(Q1_VECTOR, question_id=1)

    assert match.similarity == pytest.approx(1.0, abs=1e-5)


async def test_orthogonal_vector_scores_zero(store: ModuleType) -> None:
    await store.replace_anchors([_anchor("q1-a0", 1, Q1_VECTOR)])

    match = await store.match(Q2_VECTOR, question_id=1)

    assert match.similarity == pytest.approx(0.0, abs=1e-5)


async def test_question_filter_separates_own_and_rival(store: ModuleType) -> None:
    """similarity는 해당 문제, rival_similarity는 나머지 전체에서 나와야 한다."""
    await store.replace_anchors(
        [_anchor("q1-a0", 1, Q1_VECTOR), _anchor("q2-a0", 2, Q2_VECTOR)]
    )

    match = await store.match(Q1_VECTOR, question_id=1)

    assert match.similarity == pytest.approx(1.0, abs=1e-5)
    assert match.rival_similarity == pytest.approx(0.0, abs=1e-5)


async def test_ambiguous_vector_has_narrow_margin(store: ModuleType) -> None:
    """두 문제 정답 사이에 걸친 답변은 margin이 좁게 나온다.

    "몰라요" 같은 답변을 거르는 근거가 이 수치다.
    """
    await store.replace_anchors(
        [_anchor("q1-a0", 1, Q1_VECTOR), _anchor("q2-a0", 2, Q2_VECTOR)]
    )

    match = await store.match(DIAGONAL, question_id=1)

    assert match.similarity == pytest.approx(0.7071, abs=1e-3)
    assert match.rival_similarity == pytest.approx(0.7071, abs=1e-3)
    assert abs(match.similarity - match.rival_similarity) < settings.min_margin


async def test_takes_best_among_multiple_anchors(store: ModuleType) -> None:
    """한 문제에 기대 정답이 여럿이면 그중 최고값을 쓴다."""
    await store.replace_anchors(
        [_anchor("q1-a0", 1, Q2_VECTOR), _anchor("q1-a1", 1, Q1_VECTOR)]
    )

    match = await store.match(Q1_VECTOR, question_id=1)

    assert match.similarity == pytest.approx(1.0, abs=1e-5)


async def test_missing_question_returns_none(store: ModuleType) -> None:
    """앵커가 없는 문제는 유사도가 아니라 None이다. 0점과 구분되어야 한다."""
    await store.replace_anchors([_anchor("q1-a0", 1, Q1_VECTOR)])

    match = await store.match(Q1_VECTOR, question_id=99)

    assert match.similarity is None
    assert match.rival_similarity == pytest.approx(1.0, abs=1e-5)


async def test_single_question_has_no_rival(store: ModuleType) -> None:
    await store.replace_anchors([_anchor("q1-a0", 1, Q1_VECTOR)])

    match = await store.match(Q1_VECTOR, question_id=1)

    assert match.rival_similarity is None


async def test_judge_pipeline_on_real_store(
    store: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """판정 파이프라인을 실제 스토어에 붙여 한 바퀴 돌린다.

    임베딩만 목이고 유사도 계산부터 임계값 판정까지는 진짜 경로다.
    """
    await store.replace_anchors(
        [_anchor("q1-a0", 1, Q1_VECTOR), _anchor("q2-a0", 2, Q2_VECTOR)]
    )

    async def fake_embed_one(text: str) -> list[float]:
        return Q1_VECTOR

    async def forbidden(*args: object, **kwargs: object) -> bool:
        raise AssertionError("명확한 정답에 LLM을 부르면 안 된다")

    monkeypatch.setattr(embedding, "embed_one", fake_embed_one)
    monkeypatch.setattr(llm, "judge_answer", forbidden)

    result = await judge_service.judge(_question(), "서울")

    assert result.is_correct is True
    assert result.judge_method == JudgeMethod.EMBEDDING
    assert result.similarity == pytest.approx(1.0, abs=1e-5)
    assert result.rival_similarity == pytest.approx(0.0, abs=1e-5)


async def test_judge_sends_ambiguous_answer_to_llm_on_real_store(
    store: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """두 문제 사이에 걸친 답변은 실제 스토어에서도 LLM으로 넘어간다."""
    await store.replace_anchors(
        [_anchor("q1-a0", 1, Q1_VECTOR), _anchor("q2-a0", 2, Q2_VECTOR)]
    )
    calls = {"llm": 0}

    async def fake_embed_one(text: str) -> list[float]:
        return DIAGONAL

    async def fake_judge(*args: object, **kwargs: object) -> bool:
        calls["llm"] += 1
        return False

    monkeypatch.setattr(embedding, "embed_one", fake_embed_one)
    monkeypatch.setattr(llm, "judge_answer", fake_judge)

    result = await judge_service.judge(_question(), "몰라요")

    assert result.judge_method == JudgeMethod.LLM
    assert result.is_correct is False
    assert calls["llm"] == 1


async def test_replace_clears_previous_anchors(store: ModuleType) -> None:
    """재적재는 통째로 갈아끼운다.

    모델이나 템플릿을 바꿔 다시 넣을 때 옛 벡터가 남으면
    비교가 성립하지 않는 값끼리 섞인다.
    """
    await store.replace_anchors([_anchor("q1-a0", 1, Q1_VECTOR)])
    await store.replace_anchors([_anchor("q2-a0", 2, Q2_VECTOR)])

    match = await store.match(Q1_VECTOR, question_id=1)

    assert match.similarity is None
