"""임베딩 클라이언트 테스트.

여기서 보는 것은 임베딩 품질이 아니라 **응답을 입력에 다시 붙이는 과정**이다.
이 대응이 어긋나면 앵커에 남의 벡터가 저장되는데, 개수는 맞으므로 어디서도
예외가 나지 않고 판정 품질만 조용히 무너진다. 그래서 여기에 테스트를 둔다.
"""

from types import SimpleNamespace

import pytest

from app.core import embedding
from app.core.exceptions import EmbeddingUnavailableError


def _response(pairs: list[tuple[int, list[float]]]) -> SimpleNamespace:
    """(index, embedding) 목록으로 SDK 응답을 흉내낸다."""
    return SimpleNamespace(
        data=[SimpleNamespace(index=i, embedding=vec) for i, vec in pairs]
    )


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """게이트와 SDK를 걷어내고 응답만 갈아끼운다."""
    state = SimpleNamespace(response=None, calls=0)

    async def fake_call_guarded(gate, kind, call):  # noqa: ANN001, ARG001
        state.calls += 1
        return state.response

    monkeypatch.setattr(embedding, "call_guarded", fake_call_guarded)
    return state


@pytest.mark.asyncio
async def test_뒤섞인_응답을_index로_되돌린다(capture: SimpleNamespace) -> None:
    """SDK가 순서를 바꿔 돌려줘도 입력 순서대로 나와야 한다.

    응답 항목에 index 필드가 있다는 것 자체가 배열 순서를 믿으면 안 된다는 뜻이다.
    """
    capture.response = _response([(2, [3.0]), (0, [1.0]), (1, [2.0])])

    vectors = await embedding.embed(["첫째", "둘째", "셋째"])

    assert vectors == [[1.0], [2.0], [3.0]]


@pytest.mark.asyncio
async def test_정상_순서_응답도_그대로_통과한다(capture: SimpleNamespace) -> None:
    capture.response = _response([(0, [1.0]), (1, [2.0])])

    assert await embedding.embed(["첫째", "둘째"]) == [[1.0], [2.0]]


@pytest.mark.asyncio
async def test_응답_개수가_모자라면_예외(capture: SimpleNamespace) -> None:
    """조용히 짧은 리스트를 돌려주면 호출부의 zip이 어긋난다. 여기서 끊는다."""
    capture.response = _response([(0, [1.0])])

    with pytest.raises(EmbeddingUnavailableError, match="개수"):
        await embedding.embed(["첫째", "둘째"])


@pytest.mark.asyncio
async def test_빈_입력은_호출하지_않는다(capture: SimpleNamespace) -> None:
    """빈 배열로 부르면 400이 난다. 부르기 전에 끊는 것이 맞다."""
    assert await embedding.embed([]) == []
    assert capture.calls == 0
