"""흐름 정형 테스트 (세마포어와 토큰 버킷).

시간을 주입해서 실제로 기다리지 않는다. 실시간에 기대는 테스트는
느리고 CI에서 불안정하다.
"""

import asyncio

import pytest

from app.core import metrics
from app.core.config import settings
from app.core.rate_limit import OutboundGate, TokenBucket


class FakeClock:
    """가상 시계. sleep이 실제 대기 대신 시계를 앞으로 돌린다."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


# --- 토큰 버킷 ---


async def test_bucket_allows_first_call_immediately() -> None:
    clock = FakeClock()
    bucket = TokenBucket(60, clock=clock, sleep=clock.sleep)

    waited = await bucket.acquire()

    assert waited == 0.0


async def test_bucket_waits_when_tokens_run_out() -> None:
    """분당 60회면 초당 1회. 버킷이 비면 다음 토큰까지 1초를 기다린다."""
    clock = FakeClock()
    bucket = TokenBucket(60, clock=clock, sleep=clock.sleep)

    await bucket.acquire()
    waited = await bucket.acquire()

    assert waited == pytest.approx(1.0)
    assert clock.now == pytest.approx(1.0)


async def test_bucket_refills_over_time() -> None:
    """시간이 지나면 다시 즉시 통과한다."""
    clock = FakeClock()
    bucket = TokenBucket(60, clock=clock, sleep=clock.sleep)
    await bucket.acquire()

    clock.now += 5.0
    waited = await bucket.acquire()

    assert waited == 0.0


async def test_bucket_allows_burst_up_to_capacity() -> None:
    """순간 몰림은 capacity까지 허용한다. 그 다음부터 기다린다."""
    clock = FakeClock()
    bucket = TokenBucket(600, capacity=10, clock=clock, sleep=clock.sleep)

    for _ in range(10):
        assert await bucket.acquire() == 0.0

    assert await bucket.acquire() > 0.0


async def test_bucket_does_not_accumulate_beyond_capacity() -> None:
    """오래 쉬었다고 무한정 쌓이지 않는다. 쌓이면 한 번에 쿼터를 터뜨린다.

    이 테스트가 실제 무한 루프를 잡았다. 시각이 3600 근처로 커지면
    3600.1 - 3600.0 이 0.09999999999990905 가 되어 토큰이 1에 미세하게
    못 미치고, 그때 계산된 대기 시간이 시계 해상도보다 작아 상태가
    영원히 변하지 않았다. acquire의 오차 허용치와 최소 대기가 그 대응이다.
    """
    clock = FakeClock()
    bucket = TokenBucket(600, capacity=10, clock=clock, sleep=clock.sleep)

    clock.now += 3600.0  # 한 시간 방치

    for _ in range(10):
        assert await bucket.acquire() == 0.0
    assert await bucket.acquire() > 0.0


@pytest.mark.parametrize("start_time", [0.0, 3600.0, 86_400.0, 1_000_000.0])
async def test_bucket_makes_progress_at_any_clock_offset(start_time: float) -> None:
    """시계가 얼마나 오래 돌았든 진행해야 한다.

    monotonic 시계는 부팅 이후로 계속 커지므로, 오래 켜둔 서버에서는
    큰 값끼리 빼게 된다. 그 자리에서 정밀도가 떨어진다.
    """
    clock = FakeClock()
    clock.now = start_time
    bucket = TokenBucket(600, capacity=2, clock=clock, sleep=clock.sleep)

    for _ in range(6):
        await bucket.acquire()  # 멈추지 않고 끝나는 것 자체가 단정이다


async def test_bucket_rejects_nonpositive_rate() -> None:
    with pytest.raises(ValueError):
        TokenBucket(0)


# --- 게이트 (세마포어) ---


async def _peak_concurrency(gate: OutboundGate, workers: int) -> int:
    current = 0
    peak = 0

    async def worker() -> None:
        nonlocal current, peak
        async with gate.hold():
            current += 1
            peak = max(peak, current)
            await asyncio.sleep(0.01)
            current -= 1

    await asyncio.gather(*[worker() for _ in range(workers)])
    return peak


async def test_gate_caps_in_flight_requests() -> None:
    gate = OutboundGate("test", max_concurrency=2, rate_per_minute=100_000)

    assert await _peak_concurrency(gate, workers=10) == 2


async def test_gate_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """설정만 뒤집어 끌 수 있어야 부하 실험에서 대응 전후를 같은 코드로 잰다."""
    monkeypatch.setattr(settings, "gate_enabled", False)
    gate = OutboundGate("test", max_concurrency=2, rate_per_minute=100_000)

    assert await _peak_concurrency(gate, workers=10) > 2


async def test_gate_records_throttle_wait() -> None:
    """레이트 리미터에 걸려 기다린 시간이 계측에 남는다.

    이 숫자가 없으면 "느려진 이유가 우리 리미터인지 제공사인지"를 구분할 수 없다.
    """
    # 분당 6000회 = 초당 100회, 버스트 100. 101번째부터 기다린다
    gate = OutboundGate("test", max_concurrency=10, rate_per_minute=6000)

    for _ in range(102):
        async with gate.hold():
            pass

    waits = metrics.snapshot()["waits"]
    assert waits["test.throttle_wait_seconds"]["count"] >= 1
