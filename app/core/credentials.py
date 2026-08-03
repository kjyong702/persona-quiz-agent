"""자격증명을 다시 읽는다.

**설정을 두 부류로 나누는 것이 이 모듈의 출발점이다.**

| 부류 | 예 | 바뀌면 |
|---|---|---|
| **동작을 바꾸는 설정** | 모델 ID, 임계값, 프롬프트 | **재배포해야 한다.** 평가를 다시 돌려야 하기 때문이다 |
| **동작을 안 바꾸는 자격증명** | API 키 | **무중단으로 갱신되어야 한다** |

앞은 프로세스 수명 내내 고정하는 것이 오히려 옳다. 임계값이 도중에 바뀌면
같은 답변이 앞뒤로 다르게 판정되고, 그 사실이 어디에도 안 남는다.

**뒤는 다르다.** 키가 바뀌어도 판정 결과는 같아야 한다. 그런데 지금 구조는
둘을 구분하지 않는다. `Settings()`가 부팅 시 한 번 `.env`와 환경변수를 읽고,
클라이언트는 첫 호출에 그 값을 박아 프로세스 내내 재사용한다.
**키를 회전시켜도 재배포 전까지 반영되지 않는다.**

## 왜 TTL만으로는 부족한가

주기적으로 클라이언트를 새로 만들면 **HTTP 커넥션 풀이 매번 버려진다.**
`AsyncOpenAI`는 내부에 `httpx` 클라이언트를 들고 연결을 재사용하는데,
재생성하면 그 연결이 끊기고 다음 호출들이 TLS 핸드셰이크를 다시 한다.

그래서 **TTL이 지나면 값을 다시 읽되, 값이 실제로 달라졌을 때만 재생성한다.**
대부분의 경우 값이 같으므로 재생성이 일어나지 않는다.

## 무엇을 안 하는가

키 자체의 회전은 제공사와 시크릿 매니저의 일이다. 구 키와 신 키가 겹치는
기간을 두는 것도 그쪽이다. **이 모듈은 "회전된 값을 언제 알아채는가"만 다룬다.**
"""

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from app.core import metrics
from app.core.config import settings

T = TypeVar("T")

ENV_VAR = "OPENAI_API_KEY"


def current_api_key() -> str | None:
    """지금 이 순간의 키.

    환경변수를 먼저 본다. 컨테이너에서 시크릿을 갈아끼우면 보통 여기로 온다.
    없으면 부팅 시 읽은 설정값으로 물러난다.

    `.env` 파일을 다시 파싱하지는 않는다. 파일은 이미지에 안 들어가는 것이
    전제이고, 로컬 개발에서는 어차피 프로세스를 다시 띄우기 때문이다.
    """
    return os.environ.get(ENV_VAR) or settings.openai_api_key


def fingerprint(secret: str | None) -> str:
    """로그와 지표에 남길 수 있는 형태.

    **키 자체는 절대 남기지 않는다.** 앞 8자만 쓰는 것은 회전을 구분하기에
    충분하면서 원본을 복원할 수 없기 때문이다.
    """
    if not secret:
        return "none"
    return hashlib.sha256(secret.encode()).hexdigest()[:8]


@dataclass
class RefreshableClient(Generic[T]):
    """TTL이 지나면 자격증명을 다시 읽고, 달라졌을 때만 다시 만든다.

    `embedding`과 `llm` 두 모듈이 같은 패턴을 각자 들고 있었다. 한쪽만 고치면
    다른 쪽은 옛 키로 계속 호출한다. 그래서 한군데로 모았다.
    """

    build: Callable[[str], T]
    name: str
    ttl_seconds: float = 300.0
    _client: T | None = None
    _key_hash: str = ""
    _checked_at: float = 0.0
    _loaded_at: float = 0.0

    def get(self, missing_key_error: Callable[[str], Exception]) -> T:
        now = time.monotonic()
        if self._client is not None and now - self._checked_at < self.ttl_seconds:
            return self._client

        key = current_api_key()
        if not key:
            raise missing_key_error(f"{ENV_VAR}가 설정되지 않았습니다")

        self._checked_at = now
        key_hash = fingerprint(key)
        if self._client is not None and key_hash == self._key_hash:
            # 값이 그대로다. 커넥션 풀을 살려둔다
            return self._client

        rotated = self._client is not None
        self._client = self.build(key)
        self._key_hash = key_hash
        self._loaded_at = time.time()
        metrics.increment(f"{self.name}.client.{'rotated' if rotated else 'created'}", 1)
        return self._client

    def status(self) -> dict[str, object]:
        """헬스체크가 보여줄 값. 키는 지문으로만 나간다."""
        return {
            "key_fingerprint": self._key_hash or "none",
            "loaded_at": self._loaded_at or None,
            "ttl_seconds": self.ttl_seconds,
        }

    def reset(self) -> None:
        """테스트용. 프로세스 전역 상태를 쓰는 대가다."""
        self._client = None
        self._key_hash = ""
        self._checked_at = 0.0
        self._loaded_at = 0.0
