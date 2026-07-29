"""OpenAI 임베딩 클라이언트.

서비스가 SDK를 직접 부르지 않고 이 모듈을 거친다. Phase 3.5에서 동시 인플라이트
상한과 Retry-After를 존중하는 재시도가 이 안으로 들어오는데, 호출부가 흩어져 있으면
그 제어를 한 곳에 넣을 수가 없다.
"""

from openai import APIError, AsyncOpenAI

from app.core.config import settings
from app.core.exceptions import EmbeddingUnavailableError

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        if not settings.openai_api_key:
            raise EmbeddingUnavailableError("OPENAI_API_KEY가 설정되지 않았습니다")
        _client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_seconds,
        )
    return _client


async def embed(texts: list[str]) -> list[list[float]]:
    """여러 문자열을 한 번의 호출로 임베딩한다.

    시드 적재에서 앵커 수십 개를 넣을 때 한 건씩 부르면 호출 수만큼 쿼터를 쓴다.
    배치가 기본이고 단건은 이 함수를 감싼다.
    """
    if not texts:
        return []
    try:
        response = await _get_client().embeddings.create(
            model=settings.embedding_model,
            input=texts,
        )
    except APIError as exc:
        raise EmbeddingUnavailableError(f"임베딩 호출 실패: {exc}") from exc
    return [item.embedding for item in response.data]


async def embed_one(text: str) -> list[float]:
    return (await embed([text]))[0]
