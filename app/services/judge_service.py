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
    IndexDriftError,
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
    except IndexDriftError as exc:
        # **여기는 폴백하지 않는다.** 아래 except보다 먼저 잡는 이유가 그것이다.
        # IndexDriftError도 ExternalServiceError라서 순서를 바꾸면 조용히 LLM으로
        # 넘어가고, 인덱스가 깨진 채로 서비스가 계속 돈다. 막으려던 바로 그 상황이다.
        #
        # 다른 외부 장애는 일시적이라 다른 경로로 가는 것이 맞지만 이건 구성 오류다.
        # 재시도해도 낫지 않고 사람이 다시 적재해야 풀린다
        raise ServiceUnavailableError(ErrorCode.INDEX_DRIFT, str(exc)) from exc
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
    """상한을 넘겼는가.

    **예전에는 조건이 하나 더 있었다.** 다른 문제 앵커와의 차이(margin)가 좁으면
    확정하지 않고 LLM으로 넘겼다. 근거는 "몰라요" 같은 짧고 흔한 답변이 모든 문제에
    두루 가까울 테니 차이로 걸러진다는 것이었다.

    **측정으로 두 가지가 드러나 걷어냈다** (docs/notes/threshold-measurement.md).

    1. 평가셋 372건에서 **한 번도 발화하지 않았다.** 상한 통과 131건 중 0건
    2. 전제가 틀렸다. 회피성 답변의 유사도 중앙값은 0.256으로 **상한 근처에도 못 온다.**
       하한에서 먼저 끊기므로 margin이 볼 일이 없다

    구조적인 이유가 있다. rival은 **다른 문제** 앵커 중 최고값이라, 발화하려면 한 답변이
    자기 문제와 남의 문제 앵커에 **동시에** 가까워야 한다. 지금 20문제는 주제가 서로 멀어
    (수도/행성/왕/대양) rival 중앙값이 0.371에 그친다.

    **rival은 계속 측정한다.** 판정에 쓰지 않을 뿐 기록은 남긴다. 문제가 늘어 주제가
    겹치기 시작하면 rival이 오르고, 그때가 이 조건을 되살릴 시점이다. 앵커를 58에서
    121개로 늘렸을 때 이미 0.347 -> 0.371로 움직였다.
    """
    assert match.similarity is not None
    return match.similarity >= settings.upper_threshold


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
