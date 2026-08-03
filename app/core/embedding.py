"""OpenAI 임베딩 클라이언트.

서비스가 SDK를 직접 부르지 않고 이 모듈을 거친다. Phase 3.5에서 동시 인플라이트
상한과 Retry-After를 존중하는 재시도가 이 안으로 들어오는데, 호출부가 흩어져 있으면
그 제어를 한 곳에 넣을 수가 없다.
"""

from openai import APIError, AsyncOpenAI

from app.core import credentials
from app.core.config import settings
from app.core.exceptions import EmbeddingUnavailableError
from app.core.rate_limit import embedding_gate
from app.core.retry import call_guarded

_clients = credentials.RefreshableClient[AsyncOpenAI](
    name="embedding",
    ttl_seconds=settings.credential_ttl_seconds,
    build=lambda key: AsyncOpenAI(
        api_key=key,
        timeout=settings.llm_timeout_seconds,
        # SDK 자체 재시도를 끄고 우리가 관리한다. SDK가 안에서 몰래
        # 다시 걸면 그 요청은 세마포어와 레이트 리미터를 거치지 않고 나가서
        # 흐름 제어에 구멍이 생기고, 계측 숫자도 실제 호출 수와 어긋난다
        max_retries=0,
    ),
)


def _get_client() -> AsyncOpenAI:
    """판정 쪽과 같은 이유로 자격증명을 다시 읽는다. app/core/credentials.py 참고.

    **두 모듈이 각자 클라이언트를 들고 있으므로 한쪽만 고치면 안 된다.**
    임베딩만 새 키로 가고 판정은 옛 키로 가면 절반만 살아 있는 상태가 된다.
    """
    return _clients.get(EmbeddingUnavailableError)


async def embed(texts: list[str]) -> list[list[float]]:
    """여러 문자열을 한 번의 호출로 임베딩한다.

    시드 적재에서 앵커 수십 개를 넣을 때 한 건씩 부르면 호출 수만큼 쿼터를 쓴다.
    배치가 기본이고 단건은 이 함수를 감싼다.
    """
    if not texts:
        return []

    async def _call() -> object:
        return await _get_client().embeddings.create(
            model=settings.embedding_model,
            input=texts,
        )

    try:
        response = await call_guarded(embedding_gate, "embedding", _call)
    except APIError as exc:
        raise EmbeddingUnavailableError(f"임베딩 호출 실패: {exc}") from exc

    # 응답 순서를 믿지 않고 index로 정렬한다. 응답 항목에 index 필드가 있다는 것
    # 자체가 배열 순서가 입력 순서와 같다고 보장하지 않는다는 뜻이다.
    # 여기서 순서가 어긋나면 앵커와 벡터가 뒤바뀌어 저장되는데, 개수는 맞으므로
    # zip(strict=True)도 통과하고 예외도 나지 않는다. 판정만 조용히 엉망이 된다
    items = sorted(response.data, key=lambda item: item.index)  # type: ignore[attr-defined]
    if len(items) != len(texts):
        raise EmbeddingUnavailableError(
            f"임베딩 응답 개수가 입력과 다릅니다: 입력 {len(texts)}건, 응답 {len(items)}건"
        )
    return [item.embedding for item in items]


async def embed_one(text: str) -> list[float]:
    return (await embed([text]))[0]
