"""나가는 외부 호출 계측.

부하 실험에서 대응 전후를 비교하려면 서버 안쪽 숫자가 필요하다.
클라이언트가 보는 성공률만으로는 **429가 몇 번 났는지 알 수 없다.**
재시도가 흡수해버리기 때문이다. 밖에서는 "느렸지만 성공"으로만 보인다.

프로세스 메모리에만 쌓이고 재시작하면 사라진다. 관측 도구를 붙이는 것은
이 프로젝트 범위가 아니고 여기서는 실험의 계측 지점으로만 쓴다.
"""

from collections import defaultdict
from typing import Any

_counters: defaultdict[str, int] = defaultdict(int)
_waits: defaultdict[str, list[float]] = defaultdict(list)


def increment(name: str, by: int = 1) -> None:
    _counters[name] += by


def observe_wait(name: str, seconds: float) -> None:
    _waits[name].append(seconds)


def snapshot() -> dict[str, Any]:
    return {
        "counters": dict(sorted(_counters.items())),
        "waits": {
            name: {
                "count": len(values),
                "total": round(sum(values), 3),
                "max": round(max(values), 3),
            }
            for name, values in sorted(_waits.items())
            if values
        },
    }


def reset() -> None:
    """테스트용. 카운터가 테스트 사이에 새면 단정이 서로 오염된다."""
    _counters.clear()
    _waits.clear()
