"""ChromaDB 벡터 스토어.

기대 정답(앵커)의 임베딩을 보관하고 답변 벡터와의 유사도를 준다.

임베딩은 우리가 만들어 넣는다. Chroma의 기본 임베딩 함수를 쓰지 않는 이유는
판정에 쓰는 모델을 코드에서 명시적으로 고정하기 위해서다. 스토어가 알아서
모델을 고르면 어느 모델로 잰 유사도인지가 코드에서 사라진다.

Chroma 클라이언트는 동기라 asyncio.to_thread로 감싼다. 요청 경로에서 그냥 부르면
이벤트 루프가 막히고, Phase 3.5의 동시성 제어가 무의미해진다.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.core.exceptions import VectorStoreUnavailableError

COLLECTION_NAME = "expected_answers"

_client: chromadb.ClientAPI | None = None


@dataclass(frozen=True)
class Anchor:
    """벡터 스토어에 넣을 기대 정답 한 건."""

    id: str
    embedding: list[float]
    document: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AnchorMatch:
    """답변 벡터와 앵커들의 비교 결과.

    similarity는 해당 문제의 앵커 중 최고, rival_similarity는 다른 문제의 앵커 중 최고.
    둘 다 None일 수 있다. similarity가 None이면 이 문제의 앵커가 스토어에 없다는 뜻이고,
    rival_similarity가 None이면 비교할 다른 문제가 없다는 뜻이다.
    """

    similarity: float | None
    rival_similarity: float | None


def _get_collection() -> chromadb.Collection:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=settings.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client.get_or_create_collection(
        name=COLLECTION_NAME,
        # 코사인으로 고정한다. 기본값(L2)이면 유사도 = 1 - 거리 환산이 성립하지 않는다
        metadata={"hnsw:space": "cosine"},
    )


def _replace_anchors_sync(anchors: list[Anchor]) -> None:
    # 앵커를 다시 적재할 때는 통째로 지우고 넣는다. 남아 있는 옛 벡터가
    # 다른 모델이나 다른 템플릿으로 만들어진 것이면 비교가 성립하지 않는다
    _get_collection()  # 컬렉션이 없으면 delete가 실패하므로 먼저 보장한다
    assert _client is not None
    _client.delete_collection(COLLECTION_NAME)

    collection = _get_collection()
    if not anchors:
        return
    collection.add(
        ids=[a.id for a in anchors],
        embeddings=[a.embedding for a in anchors],
        documents=[a.document for a in anchors],
        metadatas=[a.metadata for a in anchors],
    )


def _top_similarity(collection: chromadb.Collection, vector: list[float], where: dict[str, Any]) -> float | None:
    result = collection.query(query_embeddings=[vector], n_results=1, where=where)
    distances = (result.get("distances") or [[]])[0]
    if not distances:
        return None
    # 코사인 공간에서 Chroma의 거리는 1 - 코사인 유사도다
    return 1.0 - float(distances[0])


def _match_sync(vector: list[float], question_id: int) -> AnchorMatch:
    collection = _get_collection()
    return AnchorMatch(
        similarity=_top_similarity(collection, vector, {"question_id": question_id}),
        rival_similarity=_top_similarity(
            collection, vector, {"question_id": {"$ne": question_id}}
        ),
    )


async def replace_anchors(anchors: list[Anchor]) -> None:
    try:
        await asyncio.to_thread(_replace_anchors_sync, anchors)
    except Exception as exc:  # Chroma는 예외 계통이 넓어 여기서 하나로 좁힌다
        raise VectorStoreUnavailableError(f"앵커 적재 실패: {exc}") from exc


async def count() -> int:
    """스토어에 실제로 들어 있는 앵커 수.

    시드가 "적재했다"고 말한 수와 이 값을 대조하기 위한 것이다. 둘이 어긋나도
    조회는 정상 동작하므로 대조하지 않으면 누락을 알아챌 방법이 없다.
    """
    try:
        return await asyncio.to_thread(lambda: _get_collection().count())
    except Exception as exc:
        raise VectorStoreUnavailableError(f"앵커 개수 조회 실패: {exc}") from exc


async def match(vector: list[float], question_id: int) -> AnchorMatch:
    try:
        return await asyncio.to_thread(_match_sync, vector, question_id)
    except Exception as exc:
        raise VectorStoreUnavailableError(f"앵커 조회 실패: {exc}") from exc
