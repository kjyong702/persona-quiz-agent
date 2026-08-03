"""LLM 판정 클라이언트.

임베딩만으로 가릴 수 없는 답변을 판정한다. 프롬프트는 prompts/ 아래 버전 파일이고 어느 것을 쓰는지는 prompts.JUDGE_PROMPT가 정한다.
"""

import math

from openai import APIError, AsyncOpenAI

from app.core import credentials, log, metrics, prompts
from app.core.config import settings
from app.core.exceptions import LLMUnavailableError
from app.core.rate_limit import llm_gate
from app.core.retry import call_guarded

_log = log.get(__name__)

_clients = credentials.RefreshableClient[AsyncOpenAI](
    name="llm",
    ttl_seconds=settings.credential_ttl_seconds,
    build=lambda key: AsyncOpenAI(
        api_key=key,
        timeout=settings.llm_timeout_seconds,
        # SDK 자체 재시도를 끄고 우리가 관리한다 (app/core/retry.py 참고)
        max_retries=0,
    ),
)


def _get_client() -> AsyncOpenAI:
    """키가 회전되면 다음 TTL 경계에서 새 클라이언트로 바뀐다.

    예전에는 첫 호출에 만든 클라이언트를 프로세스 내내 재사용했다. 키를
    회전시켜도 재배포 전까지 반영되지 않았다. 근거는 app/core/credentials.py
    """
    return _clients.get(LLMUnavailableError)


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


def correct_probability(response: object) -> float | None:
    """첫 토큰 후보에서 correct 쪽 확률을 모은다.

    토큰 하나를 보는 것으로는 부족하다. `correct`, ` correct`, `Correct`, `cor`가
    모두 같은 판정으로 이어지는데 후보 목록에는 따로 잡히기 때문이다. 정규화한
    문자열을 키로 딕셔너리를 만들면 **뒤에 오는 낮은 확률이 앞의 높은 확률을
    덮어써서** 1위 확률이 0으로 보인다. 실제로 그렇게 재다가 결론을 정반대로
    읽을 뻔했다. 그래서 덮어쓰지 않고 더한다.
    """
    try:
        top = response.choices[0].logprobs.content[0].top_logprobs  # type: ignore[attr-defined]
    except (AttributeError, IndexError, TypeError):
        # 응답 모양이 바뀌었거나 logprobs가 안 왔다. 판정 자체는 계속되지만
        # **불안정 관측이 통째로 죽은 상태**라 지표가 0으로 보인다. 남겨야 안다
        _log.warning("judge.logprobs_unavailable", exc_info=True)
        return None
    total = 0.0
    for candidate in top:
        token = candidate.token.strip().lower()
        # incorrect가 correct를 포함하므로 incorrect를 먼저 걸러야 한다.
        # parse_verdict에서 겪은 함정이 여기서도 그대로 나온다
        if token.startswith("incorrect"):
            continue
        if token.startswith("correct") or token in ("cor", "corr"):
            total += math.exp(candidate.logprob)
    return total


def _record_stability(response: object) -> None:
    """이 판정이 다시 물었을 때 뒤집힐 자리인지 기록한다. 판정은 바꾸지 않는다."""
    probability = correct_probability(response)
    if probability is None:
        return
    if settings.unstable_low <= probability <= settings.unstable_high:
        metrics.increment("judge.unstable", 1)
    else:
        metrics.increment("judge.stable", 1)


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
            # 시드는 best-effort다. 넣는다고 결정적이 되지는 않지만, 안 넣으면
            # 제공사가 흔들림을 줄일 여지 자체가 없다. 상세는 config.judge_seed
            **({"seed": settings.judge_seed} if settings.judge_seed is not None else {}),
            # 판정이 흔들릴 자리인지 보려고 받는다. 응답이 커지지만 첫 토큰의
            # 후보 5개뿐이라 비용 영향은 없다
            logprobs=True,
            top_logprobs=5,
        )

    try:
        response = await call_guarded(llm_gate, "llm", _call)
    except APIError as exc:
        raise LLMUnavailableError(f"판정 호출 실패: {exc}") from exc

    # 제공사가 모델 가중치나 인프라 설정을 바꾸면 이 값이 바뀐다. 바뀌면 이전
    # 평가 수치와 비교할 근거가 사라지므로, 흔들림을 조사할 때 가장 먼저 볼 값이다
    fingerprint = getattr(response, "system_fingerprint", None)
    if fingerprint:
        metrics.increment(f"judge.fingerprint.{fingerprint}", 1)

    _record_stability(response)

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
