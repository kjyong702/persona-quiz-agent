"""LLM 판정 클라이언트.

임베딩만으로 가릴 수 없는 답변을 판정한다. 프롬프트는 prompts/judge.v1.txt.
"""

from openai import APIError, AsyncOpenAI

from app.core import metrics, prompts
from app.core.config import settings
from app.core.exceptions import LLMUnavailableError
from app.core.rate_limit import llm_gate
from app.core.retry import call_guarded

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        if not settings.openai_api_key:
            raise LLMUnavailableError("OPENAI_API_KEY가 설정되지 않았습니다")
        _client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_seconds,
            # SDK 자체 재시도를 끄고 우리가 관리한다 (app/core/retry.py 참고)
            max_retries=0,
        )
    return _client


def build_user_message(
    question_text: str, expected_answers: list[str], answer_text: str
) -> str:
    expected = ", ".join(expected_answers)
    return (
        f"문제: {question_text}\n"
        f"기대 정답: {expected}\n"
        f"참가자 답변: {answer_text}"
    )


def parse_verdict(raw: str) -> bool:
    """판정 응답을 불리언으로. 해석할 수 없으면 예외.

    incorrect를 먼저 본다. correct로 먼저 검사하면 incorrect가 correct를
    포함하고 있어 오답이 정답으로 뒤집힌다.
    """
    cleaned = raw.strip().lower().strip("`\"' .")
    if cleaned.startswith("incorrect"):
        return False
    if cleaned.startswith("correct"):
        return True
    raise LLMUnavailableError(f"판정 응답을 해석할 수 없습니다: {raw!r}")


async def judge_answer(
    question_text: str, expected_answers: list[str], answer_text: str
) -> bool:
    async def _call() -> object:
        return await _get_client().chat.completions.create(
            model=settings.judge_model,
            messages=[
                {"role": "system", "content": prompts.load(prompts.JUDGE_PROMPT)},
                {
                    "role": "user",
                    "content": build_user_message(
                        question_text, expected_answers, answer_text
                    ),
                },
            ],
            # 판정은 매번 같은 답이 나와야 한다. 창의성이 필요한 자리가 아니다
            temperature=0,
            max_tokens=5,
        )

    try:
        response = await call_guarded(llm_gate, "llm", _call)
    except APIError as exc:
        raise LLMUnavailableError(f"판정 호출 실패: {exc}") from exc

    content = response.choices[0].message.content  # type: ignore[attr-defined]
    if content is None:
        raise LLMUnavailableError("판정 응답이 비어 있습니다")
    return parse_verdict(content)


async def generate_host_message(system_prompt: str, user_message: str) -> str:
    """페르소나 진행 멘트.

    판정과 달리 temperature를 0으로 두지 않는다. 같은 상황에서 매번 똑같은
    문장이 나오면 진행자가 녹음기처럼 들린다. 판정은 재현성이 전부이고
    멘트는 자연스러움이 전부라, **같은 모델을 쓰지만 설정이 반대**다.

    캐시 적중 토큰을 계측한다. 시스템 프롬프트의 앞 구간이 요청마다 같으면
    제공사가 그 부분을 재사용하고 응답에 얼마나 재사용했는지 알려준다.
    **0이 나오면 프리픽스가 흔들리고 있다는 뜻**이라 그 자체가 신호다.
    """

    async def _call() -> object:
        return await _get_client().chat.completions.create(
            model=settings.judge_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=settings.host_temperature,
            max_tokens=settings.host_max_tokens,
        )

    try:
        response = await call_guarded(llm_gate, "llm", _call)
    except APIError as exc:
        raise LLMUnavailableError(f"진행 멘트 호출 실패: {exc}") from exc

    usage = getattr(response, "usage", None)
    if usage is not None:
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0
        metrics.increment("host.prompt_tokens", usage.prompt_tokens)
        metrics.increment("host.cached_tokens", cached)

    content = response.choices[0].message.content  # type: ignore[attr-defined]
    if not content or not content.strip():
        raise LLMUnavailableError("진행 멘트가 비어 있습니다")
    return content.strip()
