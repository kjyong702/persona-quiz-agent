"""하이브리드 판정 파이프라인 테스트.

임계값 경계와 세 경로(embedding / llm / fallback)의 분기를 본다.
외부 API는 전부 목이다. 실제 호출 품질은 Phase 5 promptfoo가 맡는다.

**LLM 호출 횟수를 매번 확인한다.** 이 구조의 존재 이유가 "명확한 구간에서
LLM을 부르지 않는 것"이라, 판정 결과만 맞고 호출이 나가면 실패한 것이다.
"""

from types import SimpleNamespace

import pytest

from app.core import embedding, llm, normalization, vector_store
from app.core.config import settings
from app.core.exceptions import (
    EmbeddingUnavailableError,
    ErrorCode,
    LLMUnavailableError,
    ServiceUnavailableError,
    VectorStoreUnavailableError,
)
from app.models import JudgeMethod, Question
from app.services import judge_service

UPPER = settings.upper_threshold
LOWER = settings.lower_threshold
MARGIN = settings.min_margin


@pytest.fixture
def question() -> Question:
    return Question(
        id=1,
        quiz_set_id=1,
        order_no=1,
        question_text="대한민국의 수도는 어디인가요?",
        expected_answers=["서울", "서울특별시"],
    )


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """임베딩, 벡터 스토어, LLM을 전부 목으로 갈아끼운다."""
    state = SimpleNamespace(
        similarity=0.9,
        rival=0.1,
        embed_error=None,
        store_error=None,
        llm_verdict=True,
        llm_error=None,
    )
    calls = SimpleNamespace(llm=0, embed=0)

    async def fake_embed_one(text: str) -> list[float]:
        calls.embed += 1
        if state.embed_error is not None:
            raise state.embed_error
        return [0.1, 0.2, 0.3]

    async def fake_match(vector: list[float], question_id: int) -> vector_store.AnchorMatch:
        if state.store_error is not None:
            raise state.store_error
        return vector_store.AnchorMatch(
            similarity=state.similarity, rival_similarity=state.rival
        )

    async def fake_judge_answer(
        question_text: str, expected: list[str], answer_text: str
    ) -> bool:
        calls.llm += 1
        if state.llm_error is not None:
            raise state.llm_error
        return state.llm_verdict

    monkeypatch.setattr(embedding, "embed_one", fake_embed_one)
    monkeypatch.setattr(vector_store, "match", fake_match)
    monkeypatch.setattr(llm, "judge_answer", fake_judge_answer)
    return SimpleNamespace(state=state, calls=calls)


# --- 임베딩 단독 판정 (LLM 호출 없음) ---


async def test_above_upper_is_correct_without_llm(
    question: Question, stub: SimpleNamespace
) -> None:
    stub.state.similarity = UPPER + 0.05
    stub.state.rival = 0.2

    result = await judge_service.judge(question, "서울이요")

    assert result.is_correct is True
    assert result.judge_method == JudgeMethod.EMBEDDING
    assert stub.calls.llm == 0


async def test_exactly_upper_is_correct(
    question: Question, stub: SimpleNamespace
) -> None:
    """상한은 포함이다 (>= UPPER)."""
    stub.state.similarity = UPPER
    stub.state.rival = 0.2

    result = await judge_service.judge(question, "서울")

    assert result.is_correct is True
    assert result.judge_method == JudgeMethod.EMBEDDING
    assert stub.calls.llm == 0


async def test_exactly_lower_is_incorrect(
    question: Question, stub: SimpleNamespace
) -> None:
    """하한도 포함이다 (<= LOWER)."""
    stub.state.similarity = LOWER
    stub.state.rival = 0.1

    result = await judge_service.judge(question, "부산")

    assert result.is_correct is False
    assert result.judge_method == JudgeMethod.EMBEDDING
    assert stub.calls.llm == 0


async def test_embedding_result_records_model_and_template(
    question: Question, stub: SimpleNamespace
) -> None:
    """유사도만 남기면 나중에 해석할 수 없다. 모델과 템플릿이 함께 있어야 한다."""
    stub.state.similarity = UPPER + 0.05
    stub.state.rival = 0.2

    result = await judge_service.judge(question, "서울")

    assert result.similarity == pytest.approx(UPPER + 0.05)
    assert result.rival_similarity == pytest.approx(0.2)
    assert result.embedding_model == settings.embedding_model
    assert result.template_version == normalization.TEMPLATE_VERSION


# --- 중간 구간과 margin 조건 (LLM 2차) ---


async def test_middle_band_goes_to_llm(
    question: Question, stub: SimpleNamespace
) -> None:
    stub.state.similarity = (UPPER + LOWER) / 2
    stub.state.rival = 0.1
    stub.state.llm_verdict = True

    result = await judge_service.judge(question, "우리나라 수도")

    assert result.judge_method == JudgeMethod.LLM
    assert result.is_correct is True
    assert stub.calls.llm == 1


async def test_just_below_upper_goes_to_llm(
    question: Question, stub: SimpleNamespace
) -> None:
    stub.state.similarity = UPPER - 0.001
    stub.state.rival = 0.1

    result = await judge_service.judge(question, "서울 쪽")

    assert result.judge_method == JudgeMethod.LLM
    assert stub.calls.llm == 1


async def test_high_similarity_with_narrow_margin_goes_to_llm(
    question: Question, stub: SimpleNamespace
) -> None:
    """상한을 넘겨도 다른 문제 정답과 붙어 있으면 즉시 판정하지 않는다.

    "몰라요" 같은 답변이 모든 문제에서 비슷하게 높은 유사도를 받는 경우가 이것이다.
    절대 임계값만 두면 그대로 정답 처리된다.
    """
    stub.state.similarity = UPPER + 0.1
    stub.state.rival = UPPER + 0.1 - (MARGIN / 2)
    stub.state.llm_verdict = False

    result = await judge_service.judge(question, "몰라요")

    assert result.judge_method == JudgeMethod.LLM
    assert result.is_correct is False
    assert stub.calls.llm == 1


async def test_margin_exactly_at_threshold_is_confident(
    question: Question, stub: SimpleNamespace
) -> None:
    """margin 하한도 포함이다 (>= MIN_MARGIN)."""
    stub.state.similarity = UPPER + 0.1
    stub.state.rival = UPPER + 0.1 - MARGIN

    result = await judge_service.judge(question, "서울")

    assert result.judge_method == JudgeMethod.EMBEDDING
    assert stub.calls.llm == 0


async def test_no_rival_skips_margin_check(
    question: Question, stub: SimpleNamespace
) -> None:
    """비교할 다른 문제가 없으면 margin 조건을 적용할 근거가 없다."""
    stub.state.similarity = UPPER + 0.05
    stub.state.rival = None

    result = await judge_service.judge(question, "서울")

    assert result.judge_method == JudgeMethod.EMBEDDING
    assert result.is_correct is True
    assert stub.calls.llm == 0


# --- 폴백 ---


@pytest.mark.parametrize(
    "failure",
    [
        EmbeddingUnavailableError("임베딩 다운"),
        VectorStoreUnavailableError("스토어 다운"),
    ],
)
async def test_external_failure_falls_back_to_llm(
    question: Question, stub: SimpleNamespace, failure: Exception
) -> None:
    """외부 하나가 끊겨도 퀴즈 진행은 멈추지 않는다."""
    if isinstance(failure, EmbeddingUnavailableError):
        stub.state.embed_error = failure
    else:
        stub.state.store_error = failure
    stub.state.llm_verdict = True

    result = await judge_service.judge(question, "서울")

    assert result.judge_method == JudgeMethod.FALLBACK
    assert result.is_correct is True
    assert stub.calls.llm == 1


async def test_fallback_records_no_similarity(
    question: Question, stub: SimpleNamespace
) -> None:
    """유사도를 재지 못했으므로 모델과 템플릿도 남기지 않는다.

    값이 없는 것과 0인 것은 다르다. 폴백 판정을 임계값 튜닝 데이터에
    섞으면 안 되고, 그 구분이 이 NULL로 이루어진다.
    """
    stub.state.embed_error = EmbeddingUnavailableError("다운")

    result = await judge_service.judge(question, "서울")

    assert result.similarity is None
    assert result.rival_similarity is None
    assert result.embedding_model is None
    assert result.template_version is None


async def test_missing_anchors_falls_back(
    question: Question, stub: SimpleNamespace
) -> None:
    """이 문제의 앵커가 스토어에 없으면 비교 축이 없다. 시드 임베딩 미실행 상태."""
    stub.state.similarity = None
    stub.state.rival = 0.3

    result = await judge_service.judge(question, "서울")

    assert result.judge_method == JudgeMethod.FALLBACK
    assert stub.calls.llm == 1


# --- 양쪽 다 실패 ---


async def test_both_paths_down_raises_service_unavailable(
    question: Question, stub: SimpleNamespace
) -> None:
    """임의로 오답 처리하지 않는다. 틀린 판정이 조용히 데이터에 남는 것이 더 나쁘다."""
    stub.state.embed_error = EmbeddingUnavailableError("임베딩 다운")
    stub.state.llm_error = LLMUnavailableError("LLM 다운")

    with pytest.raises(ServiceUnavailableError) as exc_info:
        await judge_service.judge(question, "서울")

    assert exc_info.value.code == ErrorCode.JUDGE_UNAVAILABLE
    assert exc_info.value.status_code == 503


async def test_llm_failure_in_middle_band_raises(
    question: Question, stub: SimpleNamespace
) -> None:
    stub.state.similarity = (UPPER + LOWER) / 2
    stub.state.llm_error = LLMUnavailableError("LLM 다운")

    with pytest.raises(ServiceUnavailableError) as exc_info:
        await judge_service.judge(question, "서울")

    assert exc_info.value.code == ErrorCode.JUDGE_UNAVAILABLE


# --- 부정 표현 라우팅 (D안) ---------------------------------------------------
#
# 임베딩이 상한을 넘겨 정답을 확신하더라도 부정 표현이 보이면 LLM으로 넘긴다.
# 바이인코더는 "X다"와 "X 아니다"를 못 가르기 때문이다.
# **여기서 확인하는 것은 판정 결과가 아니라 어느 경로로 갔는가다.**


@pytest.mark.asyncio
async def test_부정_표현은_상한을_넘겨도_LLM으로_간다(
    question: Question, stub: SimpleNamespace
) -> None:
    stub.state.similarity = 0.95  # 상한을 한참 넘긴다
    stub.state.llm_verdict = False

    result = await judge_service.judge(question, "서울 아닙니다")

    assert result.judge_method == JudgeMethod.LLM
    assert stub.calls.llm == 1
    assert result.is_correct is False
    # 유사도는 그대로 남긴다. 왜 LLM으로 갔는지 사후에 알 수 있어야 한다
    assert result.similarity == 0.95


@pytest.mark.asyncio
async def test_받침이_붙은_활용형도_잡는다(
    question: Question, stub: SimpleNamespace
) -> None:
    """`아닙니다`는 문자열 `아니`를 포함하지 않는다. 어간에 받침이 붙어 음절이 바뀐다."""
    stub.state.similarity = 0.95

    result = await judge_service.judge(question, "서울 아닙니다")

    assert result.judge_method == JudgeMethod.LLM
    assert stub.calls.llm == 1


@pytest.mark.asyncio
async def test_부정_표현이_없으면_임베딩이_확정한다(
    question: Question, stub: SimpleNamespace
) -> None:
    """규칙이 정상 경로를 건드리지 않는지 본다."""
    stub.state.similarity = 0.95

    result = await judge_service.judge(question, "서울특별시")

    assert result.judge_method == JudgeMethod.EMBEDDING
    assert result.is_correct is True
    assert stub.calls.llm == 0


@pytest.mark.asyncio
async def test_기대_정답의_부정_표현에는_반응하지_않는다(
    question: Question, stub: SimpleNamespace
) -> None:
    """질문이 부정문이라 정답에도 부정 표현이 들어가는 문항이 있다.

    `없`, `않`은 규칙에서 뺐으므로 임베딩 확정 경로가 유지되어야 한다.
    """
    stub.state.similarity = 0.95

    result = await judge_service.judge(question, "공기가 없는 곳에서요")

    assert result.judge_method == JudgeMethod.EMBEDDING
    assert stub.calls.llm == 0


@pytest.mark.asyncio
async def test_상한_아래_부정_표현은_원래대로_LLM_구간이다(
    question: Question, stub: SimpleNamespace
) -> None:
    """규칙은 상한 통과분에만 적용된다. 중간 구간의 동작은 바뀌지 않는다."""
    stub.state.similarity = 0.70  # 상한과 하한 사이

    result = await judge_service.judge(question, "서울 아닙니다")

    assert result.judge_method == JudgeMethod.LLM
    assert stub.calls.llm == 1
