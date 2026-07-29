"""나가는 호출의 실패 흡수 (두 번째 층).

429는 오류가 아니라 **"지금 말고 나중에"라는 신호**다. 그래서 실패로 처리해
사용자에게 돌려주지 않고 흡수한다.

핵심은 **Retry-After를 존중하는 것**이다. 제공사가 "3초 뒤에 오라"고 알려줬는데
우리 지수 백오프가 0.5초 뒤에 다시 때리면 그 요청도 429가 되고, 그게 다시
재시도를 부른다. 헤더가 있으면 헤더가 이긴다. 없을 때만 지수 백오프 + 지터다.

지터를 넣는 이유는 동시에 429를 맞은 요청들이 **같은 시각에 함께 재시도해서
두 번째 파도를 만드는 것**을 막기 위해서다. 고정 백오프는 실패를 동기화시킨다.
"""

from collections.abc import Awaitable, Callable
from typing import TypeVar

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt
from tenacity.wait import wait_base, wait_exponential_jitter

from app.core import metrics
from app.core.config import settings
from app.core.rate_limit import OutboundGate

T = TypeVar("T")


def retry_after_seconds(exc: BaseException) -> float | None:
    """응답 헤더에서 제공사가 알려준 대기 시간을 꺼낸다.

    제공사마다 전달 방식이 다르다. 여기서 다루는 것은 헤더로 오는 형태이고,
    본문(예: 구조화된 RetryInfo)으로 주는 제공사를 붙일 때는 이 함수를 확장한다.
    HTTP-date 형식은 다루지 않고 None을 돌려 지수 백오프로 넘긴다.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None

    raw_ms = headers.get("retry-after-ms")
    if raw_ms is not None:
        try:
            return max(0.0, float(raw_ms) / 1000.0)
        except ValueError:
            pass

    raw = headers.get("retry-after")
    if raw is not None:
        try:
            return max(0.0, float(raw))
        except ValueError:
            return None

    return None


# 429로 오지만 다시 걸어도 절대 성공하지 않는 오류 코드.
# 잔액 소진이 대표적이다. 결제하기 전까지는 몇 번을 걸어도 같은 답이 온다
_TERMINAL_ERROR_CODES = frozenset({"insufficient_quota", "billing_hard_limit_reached"})


def is_quota_exhausted(exc: BaseException) -> bool:
    """상태 코드는 429인데 실은 결제 문제인가.

    같은 429여도 뜻이 완전히 다르다. 레이트 리밋은 "지금 말고 나중에"지만
    잔액 소진은 "결제하기 전까지 영원히 안 된다"다. 둘을 같이 다루면
    재시도가 무의미하게 대기 시간만 늘리고, 부하 실험에서는 결제 사고가
    레이트 리밋 문제로 집계되어 측정 자체가 거짓말이 된다.
    """
    return getattr(exc, "code", None) in _TERMINAL_ERROR_CODES


def is_retryable(exc: BaseException) -> bool:
    """다시 걸어볼 만한 실패인가.

    400이나 401은 몇 번을 다시 걸어도 같은 답이 온다. 재시도는 쿼터만 쓰고
    사용자 대기 시간만 늘린다. 잘못된 요청과 일시적 실패를 섞으면 안 된다.
    """
    if is_quota_exhausted(exc):
        return False
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500
    return False


class WaitRespectingRetryAfter(wait_base):
    """제공사가 준 대기 시간이 있으면 그것을, 없으면 지수 백오프 + 지터를 쓴다."""

    def __init__(self, fallback: wait_base, cap_seconds: float, operation: str) -> None:
        self._fallback = fallback
        self._cap = cap_seconds
        self._operation = operation

    def __call__(self, retry_state: object) -> float:
        outcome = getattr(retry_state, "outcome", None)
        exc = outcome.exception() if outcome is not None else None

        if exc is not None:
            hinted = retry_after_seconds(exc)
            if hinted is not None:
                metrics.increment(f"{self._operation}.retry_after_honored")
                # 상한은 둔다. 제공사가 매우 긴 값을 주면 요청 하나가
                # 그만큼 매달려 있게 되고 그동안 다른 요청의 자리도 막는다
                return min(hinted, self._cap)

        return self._fallback(retry_state)  # type: ignore[arg-type]


async def call_guarded(
    gate: OutboundGate,
    operation: str,
    call: Callable[[], Awaitable[T]],
) -> T:
    """게이트를 통과시키고, 일시적 실패는 흡수해서 호출한다.

    게이트는 **매 시도마다 잡고 놓는다.** 백오프로 자는 동안 세마포어 자리를
    쥐고 있으면 놀고 있는 요청이 남의 자리를 막는다. tenacity가 대기를
    시도 바깥에서 처리하므로 자는 동안에는 자리가 비어 있다.
    """

    async def _attempt() -> T:
        metrics.increment(f"{operation}.attempt")
        async with gate.hold():
            try:
                result = await call()
            except RateLimitError as exc:
                # 429 안에서 갈라 센다. 섞으면 부하 실험의 429 집계가 오염된다
                if is_quota_exhausted(exc):
                    metrics.increment(f"{operation}.quota_exhausted")
                else:
                    metrics.increment(f"{operation}.rate_limited")
                raise
        metrics.increment(f"{operation}.success")
        return result

    if not settings.retry_enabled:
        return await _attempt()

    retryer: AsyncRetrying = AsyncRetrying(
        stop=stop_after_attempt(settings.retry_max_attempts),
        wait=WaitRespectingRetryAfter(
            wait_exponential_jitter(
                initial=settings.retry_initial_delay,
                max=settings.retry_max_delay,
            ),
            cap_seconds=settings.retry_max_delay,
            operation=operation,
        ),
        retry=retry_if_exception(is_retryable),
        reraise=True,
    )

    try:
        return await retryer(_attempt)
    except Exception as exc:
        # 두 실패를 한 칸에 넣으면 측정값이 애매해진다.
        # "다시 걸어봤는데도 안 됐다"와 "애초에 다시 걸 대상이 아니었다"는
        # 대응의 효과를 판단할 때 뜻이 완전히 다르다
        if is_retryable(exc):
            metrics.increment(f"{operation}.giveup")
        else:
            metrics.increment(f"{operation}.failed")
        raise
