"""나가는 호출의 흐름 정형 (첫 번째 층).

이 서버의 병목은 우리 CPU가 아니라 제공사 쿼터다. 서버를 늘려서 푸는 문제가
아니라 **나가는 흐름을 한도에 맞추는 문제**다. 인스턴스를 늘리면 오히려
같은 쿼터를 더 빨리 소진해서 429가 늘어난다.

두 가지를 서로 다른 이유로 건다.

- **세마포어(동시 인플라이트 상한)**: 같은 순간에 제공사에 떠 있는 요청 수를 막는다.
  타임아웃과 재시도가 겹칠 때 요청이 무한히 쌓이는 것을 여기서 끊는다
- **토큰 버킷(분당 호출량)**: 시간당 총량을 막는다. 동시 요청이 1개여도
  빠르게 반복하면 RPM 한도는 넘는다. 세마포어로는 이걸 못 막는다

둘은 대체재가 아니라 서로 다른 축이다.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from app.core import metrics
from app.core.config import settings


# 부동소수점 오차 허용치와 최소 대기. 둘 다 acquire의 진행을 보장하기 위한 것이고
# 왜 필요한지는 acquire 안에 적어두었다
_TOKEN_EPSILON = 1e-9
_MIN_SLEEP = 1e-3


class TokenBucket:
    """분당 호출량 상한. **다만 실제 동작은 페이서(pacer)에 가깝다.**

    토큰이 초당 `rate_per_minute/60`개씩 차오르고 호출마다 하나를 쓴다.
    없으면 찰 때까지 기다린다.

    **capacity 기본값이 초당 충전량과 같다.** `rate_per_minute=300`이면
    capacity가 5다. 그래서 "분당 300건"이라는 이름과 달리 **한 번에 몰아서
    300건을 보낼 수 없고 초당 5건 근처로 고르게 나간다.**

        10건 연속 호출 -> 가상 시간 1.00초 (버스트 5건 + 초당 5건)

    **이것은 실수가 아니라 의도다.** 제공사 한도는 분당으로 표시되지만 실제로는
    짧은 구간의 몰림에도 429가 난다. 분당 총량만 맞추고 앞부분에 몰아 보내면
    그 순간에 걸린다. 고르게 내보내는 편이 안전하다.

    **다만 이름이 오해를 부른다.** "분당 300건 제한"으로 읽으면 버스트가 300까지
    가능하다고 생각하게 된다. 버스트를 늘리려면 `capacity`를 명시적으로 넘긴다.

    clock과 sleep을 주입받는 이유는 테스트에서 실제로 기다리지 않기 위해서다.
    실시간에 의존하는 테스트는 느리고 CI에서 불안정하다.
    """

    def __init__(
        self,
        rate_per_minute: float,
        capacity: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute은 0보다 커야 합니다")
        self._rate_per_second = rate_per_minute / 60.0
        self._capacity = capacity if capacity is not None else max(1.0, self._rate_per_second)
        self._tokens = self._capacity
        self._clock = clock
        self._sleep = sleep
        self._updated = clock()
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """토큰 하나를 얻는다. 기다린 시간(초)을 돌려준다."""
        waited = 0.0
        while True:
            async with self._lock:
                now = self._clock()
                elapsed = max(0.0, now - self._updated)
                self._updated = now
                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate_per_second)

                # 정확히 1.0이 아니라 오차만큼 여유를 두고 비교한다.
                # 큰 시각 값에서 뺄셈을 하면 결과가 미세하게 작아진다.
                # 예: 3600.1 - 3600.0 은 0.09999999999990905 라서
                # 1초를 기다려도 토큰이 0.9999999999990905 밖에 안 찬다
                if self._tokens >= 1.0 - _TOKEN_EPSILON:
                    self._tokens = max(0.0, self._tokens - 1.0)
                    return waited

                # 하한을 둔다. 남은 양이 아주 조금일 때 계산된 대기 시간이
                # 시계 해상도보다 작아지면 아무리 자도 상태가 변하지 않아
                # 루프가 진행하지 못한다
                sleep_for = max(_MIN_SLEEP, (1.0 - self._tokens) / self._rate_per_second)

            # 락을 놓고 잔다. 쥐고 자면 다른 호출이 토큰 상태를 못 읽는다
            await self._sleep(sleep_for)
            waited += sleep_for


class OutboundGate:
    """외부 호출 하나가 통과하는 관문.

    끌 수 있게 만든 이유는 부하 실험 때문이다. 대응 전후를 비교하려면
    코드를 고치지 않고 설정만 뒤집어야 같은 코드로 잰 값이 된다.
    """

    def __init__(self, name: str, max_concurrency: int, rate_per_minute: float) -> None:
        self.name = name
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._bucket = TokenBucket(rate_per_minute)

    @asynccontextmanager
    async def hold(self) -> AsyncIterator[None]:
        if not settings.gate_enabled:
            yield
            return

        # 토큰을 먼저 받고 세마포어를 잡는다.
        # 세마포어는 "제공사에 실제로 떠 있는 요청 수"를 뜻해야 하는데,
        # 토큰을 기다리는 시간은 아직 떠 있는 상태가 아니다
        waited = await self._bucket.acquire()
        if waited > 0:
            metrics.observe_wait(f"{self.name}.throttle_wait_seconds", waited)

        async with self._semaphore:
            yield


# 게이트는 엔드포인트별로 따로 둔다. 쿼터가 모델과 엔드포인트마다 따로 걸리므로
# 하나로 묶으면 임베딩이 채팅 쿼터를 잡아먹거나 그 반대가 된다
embedding_gate = OutboundGate(
    "embedding",
    settings.embedding_max_concurrency,
    settings.embedding_rpm,
)
llm_gate = OutboundGate("llm", settings.llm_max_concurrency, settings.llm_rpm)
