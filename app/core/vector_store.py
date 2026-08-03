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
            # 참고용이다. **대조에는 쓰지 않는다.**
            # get_or_create_collection은 컬렉션이 이미 있으면 넘긴 metadata를
            # 무시하므로(1.5.9에서 확인), 도장을 넣는 코드를 나중에 추가하면
            # 기존 인덱스에는 영영 안 찍힌다. 실제로 그렇게 만들었다가
            # 실제 인덱스에서 검사가 통과해버리는 것을 보고 알았다.
            # 대조는 레코드 metadata로 한다(_stamp_mismatch 참고)
            _STAMP_MODEL: settings.embedding_model,
            _STAMP_TEMPLATE: normalization.TEMPLATE_VERSION,
        },
    )


def _stamp_mismatch(collection: chromadb.Collection) -> tuple[str, str] | None:
    """인덱스 도장이 지금 설정과 어긋나면 (기대, 실제)를 준다.

    **컬렉션 metadata가 아니라 레코드에서 읽는다.** 처음에는 컬렉션 metadata에
    도장을 찍었는데 작동하지 않았다. Chroma의 `get_or_create_collection`은
    컬렉션이 이미 있으면 넘긴 metadata를 무시하므로, **도장을 넣는 코드를 나중에
    추가하면 기존 인덱스에는 영영 안 찍힌다.** 단위 테스트는 매번 새 컬렉션을
    만들어서 이걸 못 잡았고, 실제 인덱스로 돌려보고 나서야 드러났다.

    레코드 metadata는 `replace_anchors`가 매번 새로 쓰므로 항상 최신이다.
    그리고 **이 필드들은 처음부터 있었다.** 기록만 하고 아무도 안 읽었을 뿐이다.

    레코드가 없으면 비교할 대상이 없다. 그건 드리프트가 아니라 미적재이고
    `/readyz`가 따로 본다.
    """
    result = collection.get(limit=1, include=["metadatas"])
    metadatas = result.get("metadatas") or []
    if not metadatas:
        return None

    stored = metadatas[0] or {}
    expected = f"{settings.embedding_model}/{normalization.TEMPLATE_VERSION}"
    model = stored.get(_STAMP_MODEL)
    template = stored.get(_STAMP_TEMPLATE)
    if model is None and template is None:
        # 도장이 없는 레코드다. 이 검사가 생기기 전 형식이라 판단할 근거가 없다
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
    # **도장은 저장소가 찍는다.** 호출자에게 맡기면 어딘가는 빠진다.
    # 실제로 seed는 넣고 테스트는 안 넣어서, 테스트가 도장 없는 인덱스를 만들고
    # 드리프트 검사가 항상 통과하는 상태였다. 기억해야 지켜지는 규칙은
    # 언젠가 안 지켜진다
    stamp = {
        _STAMP_MODEL: settings.embedding_model,
        _STAMP_TEMPLATE: normalization.TEMPLATE_VERSION,
    }
    collection.add(
        ids=[a.id for a in anchors],
        embeddings=[a.embedding for a in anchors],
        documents=[a.document for a in anchors],
        metadatas=[{**a.metadata, **stamp} for a in anchors],
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


async def stamp_mismatch() -> tuple[str, str] | None:
    """헬스체크가 쓰는 도장 대조. 어긋나면 (기대, 실제)를 준다.

    `match()`는 어긋나면 예외를 던지는데 헬스체크는 던지면 안 된다.
    상태를 응답 본문에 담아야 무엇이 어긋났는지 보인다.
    """

    def _check() -> tuple[str, str] | None:
        return _stamp_mismatch(_get_collection())

    try:
        return await asyncio.to_thread(_check)
    except Exception as exc:  # 스토어 자체가 죽었으면 드리프트로 보고하지 않는다
        raise VectorStoreUnavailableError(f"도장 조회 실패: {exc}") from exc
