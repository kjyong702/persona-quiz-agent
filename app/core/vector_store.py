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

from app.core import normalization
from app.core.config import settings
from app.core.exceptions import IndexDriftError, VectorStoreUnavailableError

COLLECTION_NAME = "expected_answers"

# 인덱스에 찍는 도장. 이 값들이 지금 설정과 다르면 저장된 벡터와 질의 벡터가
# 서로 다른 공간에 있다는 뜻이라 유사도 비교가 성립하지 않는다.
# 업계에서는 이 현상을 index drift라고 부른다
_STAMP_MODEL = "embedding_model"
_STAMP_TEMPLATE = "template_version"

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
        metadata={
            "hnsw:space": "cosine",
            # 어떤 모델과 어떤 정규화 템플릿으로 만든 인덱스인지 남긴다.
            # **이미 있는 컬렉션에는 이 값이 반영되지 않는다.** Chroma의
            # get_or_create_collection은 컬렉션이 있으면 넘긴 metadata를 무시하고
            # 기존 것을 그대로 준다(1.5.9에서 확인). 그래서 이 도장은 처음
            # 만들어질 때의 값으로 고정되고, 그것이 바로 우리가 알고 싶은 값이다
            _STAMP_MODEL: settings.embedding_model,
            _STAMP_TEMPLATE: normalization.TEMPLATE_VERSION,
        },
    )


def _stamp_mismatch(collection: chromadb.Collection) -> tuple[str, str] | None:
    """인덱스 도장이 지금 설정과 어긋나면 (기대, 실제)를 준다.

    도장이 아예 없으면 이 검사가 생기기 전에 만든 인덱스다. 그때는 판단할 근거가
    없으므로 어긋났다고 보지 않는다. 다시 적재하면 도장이 찍힌다.
    """
    stored = collection.metadata or {}
    expected = f"{settings.embedding_model}/{normalization.TEMPLATE_VERSION}"
    model = stored.get(_STAMP_MODEL)
    template = stored.get(_STAMP_TEMPLATE)
    if model is None and template is None:
        return None
    actual = f"{model}/{template}"
    return None if actual == expected else (expected, actual)


def _require_matching_stamp(collection: chromadb.Collection) -> None:
    """어긋나면 판정을 멈춘다. 조용히 틀린 유사도를 내보내지 않는다.

    폴백해서 LLM으로 넘기지 않는 이유는 **인덱스가 깨진 것을 아무도 모르는 채로
    비용만 늘기 때문**이다. 판정이 안 되는 것보다 틀린 판정이 데이터에 쌓이는 쪽이
    나쁘다. LLM도 임베딩도 못 쓸 때 임의로 오답 처리하지 않는 것과 같은 판단이다.
    """
    mismatch = _stamp_mismatch(collection)
    if mismatch is None:
        return
    expected, actual = mismatch
    raise IndexDriftError(
        f"인덱스가 다른 설정으로 만들어졌습니다. 기대 {expected}, 실제 {actual}. "
        "scripts.seed로 다시 적재해야 합니다"
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
    _require_matching_stamp(collection)
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
    except IndexDriftError:
        # **포괄 except보다 먼저 통과시킨다.** 아래에서 같이 감싸면
        # VectorStoreUnavailableError가 되고, 호출자는 그것을 일시적 장애로 보고
        # LLM으로 폴백한다. 인덱스가 깨졌다는 사실이 두 겹에 걸쳐 지워진다
        raise
    except Exception as exc:
        raise VectorStoreUnavailableError(f"앵커 조회 실패: {exc}") from exc
