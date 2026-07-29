"""LLM 판정 클라이언트.

임베딩만으로 가릴 수 없는 답변을 판정한다. 프롬프트는 prompts/judge.v1.txt.
"""

from openai import APIError, AsyncOpenAI

from app.core import prompts
from app.core.config import settings
from app.core.exceptions import LLMUnavailableError

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        if not settings.openai_api_key:
            raise LLMUnavailableError("OPENAI_API_KEY가 설정되지 않았습니다")
        _client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_seconds,
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
    try:
        response = await _get_client().chat.completions.create(
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
    except APIError as exc:
        raise LLMUnavailableError(f"판정 호출 실패: {exc}") from exc

    content = response.choices[0].message.content
    if content is None:
        raise LLMUnavailableError("판정 응답이 비어 있습니다")
    return parse_verdict(content)
