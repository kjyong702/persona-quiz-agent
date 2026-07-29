"""OpenAI 임베딩 클라이언트.

서비스가 SDK를 직접 부르지 않고 이 모듈을 거친다. Phase 3.5에서 동시 인플라이트
상한과 Retry-After를 존중하는 재시도가 이 안으로 들어오는데, 호출부가 흩어져 있으면
그 제어를 한 곳에 넣을 수가 없다.
"""

from openai import APIError, AsyncOpenAI

from app.core.config import settings
from app.core.exceptions import EmbeddingUnavailableError
from app.core.rate_limit import embedding_gate
from app.core.retry import call_guarded

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        if not settings.openai_api_key:
            raise EmbeddingUnavailableError("OPENAI_API_KEY가 설정되지 않았습니다")
        _client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_seconds,
            # SDK 자체 재시도를 끄고 우리가 관리한다. SDK가 안에서 몰래
            # 다시 걸면 그 요청은 세마포어와 레이트 리미터를 거치지 않고 나가서
            # 흐름 제어에 구멍이 생기고, 계측 숫자도 실제 호출 수와 어긋난다
            max_retries=0,
        )
    return _client


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
    return [item.embedding for item in response.data]  # type: ignore[attr-defined]


async def embed_one(text: str) -> list[float]:
    return (await embed([text]))[0]
