"""헬스체크.

**둘을 나누는 이유가 있다.**

| | 무엇을 묻나 | 실패하면 |
|---|---|---|
| `/healthz` | 프로세스가 **살아 있나** | 오케스트레이터가 **재시작**한다 |
| `/readyz` | **일할 수 있나** | 트래픽만 안 보낸다. 재시작은 안 한다 |

하나로 합치면 둘이 섞인다. 앵커가 적재되지 않았다고 프로세스를 재시작하면
같은 상태로 다시 떠서 재시작 루프에 빠진다. **재시작으로 고쳐지지 않는 문제는
liveness가 아니다.**

`/readyz`가 인덱스 도장을 대조하는 이유가 그것이다. 모델을 바꾸고 재적재를 안 하면
유사도가 조용히 망가지는데(`docs/notes/index-drift.md`), **트래픽을 받기 전에
걸러내는 편이 낫다.** 판정마다 걸리면 사용자가 오류를 본다.
"""

from fastapi import APIRouter, Response

from app.core import credentials, log, normalization, prompts, vector_store
from app.core.config import settings

router = APIRouter(tags=["health"])
_log = log.get(__name__)


@router.get("/healthz")
async def liveness() -> dict[str, str]:
    """프로세스가 응답하는가. **의존 대상을 건드리지 않는다.**

    여기서 DB나 벡터 스토어를 확인하면 그것들이 잠깐 흔들릴 때 프로세스가
    재시작된다. 재시작해도 외부 의존은 그대로라 고쳐지지 않는다.
    """
    return {"status": "ok"}


@router.get("/readyz")
async def readiness(response: Response) -> dict[str, object]:
    """트래픽을 받아도 되는가.

    두 가지를 본다. 앵커가 있는가, 그리고 그 앵커가 지금 설정으로 만든 것인가.
    """
    checks: dict[str, object] = {}
    ready = True

    try:
        anchors = await vector_store.count()
        checks["anchors"] = anchors
        if anchors == 0:
            # 시드를 안 돌렸다. 판정이 전부 LLM 폴백으로 가므로 비용이 몇 배가 된다
            checks["anchors_ok"] = False
            ready = False
        else:
            checks["anchors_ok"] = True
    except Exception as exc:
        checks["anchors_ok"] = False
        checks["anchors_error"] = str(exc)
        ready = False

    drift = await vector_store.stamp_mismatch()
    checks["index_stamp"] = {
        "expected": f"{settings.embedding_model}/{normalization.TEMPLATE_VERSION}",
        "actual": drift[1] if drift else "match",
        "ok": drift is None,
    }
    if drift is not None:
        ready = False

    checks["judge_prompt"] = prompts.JUDGE_PROMPT
    checks["judge_model"] = settings.judge_model
    # 키 자체는 안 나간다. 지문이면 회전 반영 여부를 구분하기에 충분하다
    checks["api_key"] = credentials.fingerprint(credentials.current_api_key())
    if checks["api_key"] == "none":
        checks["api_key_ok"] = False
        ready = False

    if not ready:
        response.status_code = 503
        _log.error("readyz.not_ready", checks=checks)
    return {"ready": ready, "checks": checks}
