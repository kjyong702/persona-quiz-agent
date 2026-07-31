"""하이브리드 판정 파이프라인. 이 프로젝트의 핵심이다.

세 갈래로 나뉜다.

- **embedding**: 임베딩 유사도만으로 확정. LLM을 부르지 않는다
- **llm**: 임베딩으로 못 가른 구간을 LLM이 판정
- **fallback**: 임베딩 경로 자체가 실패해 LLM이 단독 판정. 유사도는 남지 않는다

설계 원칙은 "LLM을 언제 부르지 않을 수 있는가"부터 정하는 것이다.
전부 LLM에 보내면 비용이 답변 수에 비례하고 같은 답변에도 판정이 흔들린다.
명확한 구간을 임베딩으로 끊어내고 애매한 구간만 넘기는 것이 이 구조의 이유다.
"""

from app.core import embedding, llm, negation, normalization, vector_store
from app.core.config import settings
from app.core.exceptions import (
    ErrorCode,
    ExternalServiceError,
    LLMUnavailableError,
    ServiceUnavailableError,
)
from app.models import JudgeMethod, Question
from app.schemas.judge import JudgeResult


async def judge(question: Question, answer_text: str) -> JudgeResult:
    if not normalization.render(answer_text):
        # 정규화를 거치면 빈 문자열이 되는 답변이다(".", "...", "?" 등).
        # API의 min_length=1은 통과하지만 render가 끝 문장부호를 지우면 아무것도 안 남는다.
        #
        # 이걸 그대로 임베딩에 보내면 400이 나고, 그 400이 ExternalServiceError로
        # 포장되어 아래 except에 잡힌다. 결과는 같은 폴백이지만 **원인이 외부 장애로
        # 기록된다.** 제공사는 멀쩡한데 우리가 못 보낼 입력을 보낸 것이다.
        # 실패율 지표를 오염시키고, 절대 성공할 수 없는 호출에 쿼터를 쓴다
        return await _judge_by_llm(question, answer_text, JudgeMethod.FALLBACK)

    try:
        match = await _match_anchors(answer_text, question.id)
    except ExternalServiceError:
        # 임베딩이나 벡터 조회가 죽어도 판정 자체는 계속되어야 한다.
        # 외부 하나가 끊겼다고 퀴즈 진행이 멈추면 그게 더 큰 실패다
        return await _judge_by_llm(question, answer_text, JudgeMethod.FALLBACK)

    if match.similarity is None:
        # 이 문제의 앵커가 스토어에 없다. 시드 임베딩을 돌리지 않았거나
        # 문제가 나중에 추가된 경우다. 비교할 축이 없으니 LLM에 맡긴다
        return await _judge_by_llm(question, answer_text, JudgeMethod.FALLBACK)

    if _is_confident_correct(match):
        if negation.has_negation(answer_text):
            # 임베딩은 정답이라고 확신했지만 부정 표현이 보인다.
            # 바이인코더는 "X다"와 "X 아니다"를 못 가른다. 실측에서 부정문 오답이
            # 유사도 0.926까지 나왔고 정답들보다도 높았다.
            #
            # 여기서 오답으로 단정하지 않는 이유는 반어법과 전언 때문이다.
            # "아니라던데요"는 남의 말을 옮긴 것이라 정답일 수도 있다.
            # 규칙은 경로만 바꾸고 판정은 LLM이 한다
            return await _judge_by_llm(question, answer_text, JudgeMethod.LLM, match=match)
        return _embedding_result(is_correct=True, match=match)

    if match.similarity <= settings.lower_threshold:
        return _embedding_result(is_correct=False, match=match)

    # 남은 것은 애매한 구간. 여기만 LLM이 본다
    return await _judge_by_llm(question, answer_text, JudgeMethod.LLM, match=match)


def _is_confident_correct(match: vector_store.AnchorMatch) -> bool:
    """상한을 넘겼고 다른 문제 정답과도 충분히 벌어졌는가.

    상한만 보면 "몰라요", "네" 같은 짧고 흔한 답변이 통과한다. 그런 답변은
    이 문제의 정답과 가까운 만큼 다른 문제의 정답과도 가까우므로,
    두 유사도의 차이를 보면 걸러진다. 차이가 좁으면 LLM으로 넘긴다.
    """
    assert match.similarity is not None
    if match.similarity < settings.upper_threshold:
        return False
    if match.rival_similarity is None:
        # 비교할 다른 문제가 없다. margin 조건을 적용할 근거가 없으므로 통과시킨다
        return True
    return (match.similarity - match.rival_similarity) >= settings.min_margin


async def _match_anchors(answer_text: str, question_id: int) -> vector_store.AnchorMatch:
    rendered = normalization.render(answer_text)
    vector = await embedding.embed_one(rendered)
    return await vector_store.match(vector, question_id)


def _embedding_result(*, is_correct: bool, match: vector_store.AnchorMatch) -> JudgeResult:
    return JudgeResult(
        is_correct=is_correct,
        judge_method=JudgeMethod.EMBEDDING,
        similarity=match.similarity,
        rival_similarity=match.rival_similarity,
        embedding_model=settings.embedding_model,
        template_version=normalization.TEMPLATE_VERSION,
    )


async def _judge_by_llm(
    question: Question,
    answer_text: str,
    method: str,
    match: vector_store.AnchorMatch | None = None,
) -> JudgeResult:
    try:
        is_correct = await llm.judge_answer(
            question.question_text, question.expected_answers, answer_text
        )
    except LLMUnavailableError as exc:
        # 임베딩도 LLM도 못 쓰면 판정하지 않는다.
        # 임의로 오답 처리하면 틀린 판정이 조용히 데이터에 남는다
        raise ServiceUnavailableError(
            ErrorCode.JUDGE_UNAVAILABLE, "지금은 답변을 판정할 수 없습니다"
        ) from exc

    if match is None:
        # 폴백 경로. 유사도를 재지 못했으므로 모델과 템플릿도 남기지 않는다
        return JudgeResult(is_correct=is_correct, judge_method=method)

    return JudgeResult(
        is_correct=is_correct,
        judge_method=method,
        similarity=match.similarity,
        rival_similarity=match.rival_similarity,
        embedding_model=settings.embedding_model,
        template_version=normalization.TEMPLATE_VERSION,
    )
