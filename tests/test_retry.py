"""실패 흡수 테스트 (재시도, Retry-After, 백오프).

429를 목으로 주입한다. 여기서 확인하는 것은
"몇 번 다시 거는가", "얼마나 기다리는가", "무엇에는 다시 걸지 않는가"다.
"""

import asyncio

import httpx
import pytest
from openai import APIStatusError, APITimeoutError, RateLimitError
from tenacity.wait import wait_fixed

from app.core import metrics
from app.core.config import settings
from app.core.rate_limit import OutboundGate
from app.core.retry import (
    WaitRespectingRetryAfter,
    call_guarded,
    is_quota_exhausted,
    is_retryable,
    retry_after_seconds,
)


def _response(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return httpx.Response(status, headers=headers or {}, request=request)


def rate_limit_error(headers: dict[str, str] | None = None) -> RateLimitError:
    return RateLimitError("rate limited", response=_response(429, headers), body=None)


def quota_exhausted_error() -> RateLimitError:
    """잔액 소진. 상태 코드는 429지만 결제 문제다.

    실제로 받은 응답을 그대로 옮겼다 (2026-07-30, 충전 전 계정).
    """
    body = {
        "error": {
            "message": "You exceeded your current quota, please check your plan and billing details.",
            "type": "insufficient_quota",
            "param": None,
            "code": "insufficient_quota",
        }
    }
    return RateLimitError("quota", response=_response(429), body=body["error"])


def status_error(status: int) -> APIStatusError:
    return APIStatusError("boom", response=_response(status), body=None)


@pytest.fixture
def gate() -> OutboundGate:
    return OutboundGate("test", max_concurrency=4, rate_per_minute=100_000)


@pytest.fixture(autouse=True)
def fast_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """테스트에서 실제 백오프만큼 기다리지 않게 한다."""
    monkeypatch.setattr(settings, "retry_initial_delay", 0.01)
    monkeypatch.setattr(settings, "retry_max_delay", 0.01)
    monkeypatch.setattr(settings, "retry_max_attempts", 3)


# --- Retry-After 해석 ---


def test_reads_retry_after_seconds() -> None:
    assert retry_after_seconds(rate_limit_error({"retry-after": "3"})) == 3.0


def test_reads_retry_after_milliseconds() -> None:
    """밀리초 헤더가 있으면 더 정밀하므로 먼저 본다."""
    assert retry_after_seconds(rate_limit_error({"retry-after-ms": "1500"})) == 1.5


def test_returns_none_without_header() -> None:
    assert retry_after_seconds(rate_limit_error()) is None


def test_returns_none_for_http_date() -> None:
    """HTTP-date 형식은 다루지 않는다. 추측하지 않고 지수 백오프로 넘긴다."""
    header = {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}
    assert retry_after_seconds(rate_limit_error(header)) is None


def test_returns_none_for_non_api_exception() -> None:
    assert retry_after_seconds(ValueError("그냥 오류")) is None


# --- 무엇에 다시 거는가 ---


@pytest.mark.parametrize(
    "exc",
    [rate_limit_error(), status_error(500), status_error(503), APITimeoutError(request=httpx.Request("POST", "https://x"))],
)
def test_retryable_failures(exc: BaseException) -> None:
    assert is_retryable(exc) is True


def test_quota_exhaustion_is_not_retried() -> None:
    """429지만 결제 문제라 다시 걸어도 영원히 같은 답이 온다.

    실제로 겪었다. 잔액이 없는 키로 시드를 돌렸더니 insufficient_quota가
    RateLimitError(429)로 왔고, 재시도가 붙어 백오프까지 밟았다.
    """
    assert is_retryable(quota_exhausted_error()) is False


async def test_quota_exhaustion_counts_separately(gate: OutboundGate) -> None:
    """레이트 리밋 집계에 섞이면 부하 실험 결과가 거짓말이 된다."""

    async def out_of_credit() -> str:
        raise quota_exhausted_error()

    with pytest.raises(RateLimitError):
        await call_guarded(gate, "test", out_of_credit)

    counters = metrics.snapshot()["counters"]
    assert counters["test.quota_exhausted"] == 1
    assert "test.rate_limited" not in counters
    assert counters["test.attempt"] == 1  # 재시도하지 않았다


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retried(status: int) -> None:
    """400이나 401은 몇 번을 다시 걸어도 같은 답이 온다.

    재시도는 쿼터만 쓰고 사용자 대기 시간만 늘린다.
    """
    assert is_retryable(status_error(status)) is False


# --- 대기 시간 선택 ---


class _Outcome:
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def exception(self) -> BaseException:
        return self._exc


class _State:
    def __init__(self, exc: BaseException) -> None:
        self.outcome = _Outcome(exc)


def test_header_beats_exponential_backoff() -> None:
    """제공사가 알려준 시간이 있으면 그것을 쓴다.

    헤더를 무시하고 우리 백오프로 더 일찍 때리면 그 요청도 429가 되고
    그게 다시 재시도를 부른다.
    """
    wait = WaitRespectingRetryAfter(wait_fixed(99), cap_seconds=60, operation="test")

    assert wait(_State(rate_limit_error({"retry-after": "3"}))) == 3.0


def test_header_is_capped() -> None:
    """제공사가 매우 긴 값을 줘도 상한을 둔다.

    요청 하나가 그만큼 매달려 있으면 그동안 다른 요청의 자리도 막는다.
    """
    wait = WaitRespectingRetryAfter(wait_fixed(1), cap_seconds=10, operation="test")

    assert wait(_State(rate_limit_error({"retry-after": "600"}))) == 10.0


def test_falls_back_to_exponential_without_header() -> None:
    wait = WaitRespectingRetryAfter(wait_fixed(7), cap_seconds=60, operation="test")

    assert wait(_State(rate_limit_error())) == 7.0


# --- 통합: 게이트 + 재시도 ---


async def test_recovers_after_rate_limit(gate: OutboundGate) -> None:
    attempts = {"n": 0}

    async def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise rate_limit_error({"retry-after": "0"})
        return "ok"

    result = await call_guarded(gate, "test", flaky)

    assert result == "ok"
    assert attempts["n"] == 2
    counters = metrics.snapshot()["counters"]
    assert counters["test.attempt"] == 2
    assert counters["test.rate_limited"] == 1
    assert counters["test.success"] == 1
    assert counters["test.retry_after_honored"] == 1
    assert "test.giveup" not in counters


async def test_gives_up_after_max_attempts(gate: OutboundGate) -> None:
    attempts = {"n": 0}

    async def always_limited() -> str:
        attempts["n"] += 1
        raise rate_limit_error()

    with pytest.raises(RateLimitError):
        await call_guarded(gate, "test", always_limited)

    assert attempts["n"] == settings.retry_max_attempts
    counters = metrics.snapshot()["counters"]
    assert counters["test.giveup"] == 1
    assert "test.success" not in counters


async def test_does_not_retry_client_error(gate: OutboundGate) -> None:
    attempts = {"n": 0}

    async def bad_request() -> str:
        attempts["n"] += 1
        raise status_error(400)

    with pytest.raises(APIStatusError):
        await call_guarded(gate, "test", bad_request)

    assert attempts["n"] == 1


async def test_separates_giveup_from_plain_failure(gate: OutboundGate) -> None:
    """"다시 걸어봤는데도 안 됐다"와 "다시 걸 대상이 아니었다"를 섞지 않는다.

    섞으면 부하 실험에서 429 때문에 포기한 건수를 읽을 수 없다.
    """

    async def bad_request() -> str:
        raise status_error(400)

    with pytest.raises(APIStatusError):
        await call_guarded(gate, "test", bad_request)

    counters = metrics.snapshot()["counters"]
    assert counters["test.failed"] == 1
    assert "test.giveup" not in counters


async def test_retry_can_be_disabled(
    gate: OutboundGate, monkeypatch: pytest.MonkeyPatch
) -> None:
    """대응 전 상태를 재려면 재시도를 끌 수 있어야 한다."""
    monkeypatch.setattr(settings, "retry_enabled", False)
    attempts = {"n": 0}

    async def always_limited() -> str:
        attempts["n"] += 1
        raise rate_limit_error()

    with pytest.raises(RateLimitError):
        await call_guarded(gate, "test", always_limited)

    assert attempts["n"] == 1


async def test_gate_is_free_while_backing_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """백오프로 자는 동안 세마포어 자리를 쥐고 있으면 안 된다.

    놀고 있는 요청이 남의 자리를 막으면 재시도 하나가 처리량 전체를 세운다.
    자리가 하나뿐인 게이트에서 재시도 중에 다른 호출이 끼어들 수 있어야 한다.
    """
    monkeypatch.setattr(settings, "retry_initial_delay", 0.05)
    monkeypatch.setattr(settings, "retry_max_delay", 0.05)
    monkeypatch.setattr(settings, "retry_max_attempts", 3)
    gate = OutboundGate("test", max_concurrency=1, rate_per_minute=100_000)
    events: list[str] = []

    async def always_limited() -> str:
        events.append("attempt")
        raise rate_limit_error()

    async def bystander() -> None:
        await asyncio.sleep(0.015)  # 첫 시도가 끝나고 백오프에 들어간 뒤
        async with gate.hold():
            events.append("bystander")

    task = asyncio.create_task(bystander())
    with pytest.raises(RateLimitError):
        await call_guarded(gate, "test", always_limited)
    await task

    # 마지막 시도보다 앞서 끼어들었다는 것이 자리가 비어 있었다는 증거
    assert events.index("bystander") < len(events) - 1
