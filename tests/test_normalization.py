"""정규화 템플릿 테스트.

여기서 보는 것은 "같은 뜻인데 표기만 다른 문자열이 같은 형태로 모이는가"다.
이게 깨지면 유사도 차이가 의미 차이인지 표기 차이인지 구분할 수 없게 된다.
"""

import pytest

from app.core import normalization


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  서울  ", "서울"),
        ("서울 특별시", "서울 특별시"),
        ("서울\n특별시", "서울 특별시"),
        ("서울   특별시", "서울 특별시"),
        ("서울입니다.", "서울입니다"),
        ("서울!!!", "서울"),
        ("정답은 목성 ~", "정답은 목성"),
        # NFKC: 전각 문자와 호환 문자를 표준형으로
        ("ＲＧＢ", "RGB"),
        ("１년", "1년"),
    ],
)
def test_render_normalizes_notation(raw: str, expected: str) -> None:
    assert normalization.render(raw) == expected


def test_render_keeps_internal_punctuation() -> None:
    """끝의 문장부호만 지운다. 가운데 것은 의미를 가질 수 있다."""
    assert normalization.render("왓슨, 크릭.") == "왓슨, 크릭"


def test_render_keeps_case() -> None:
    """v1은 소문자화하지 않는다 (docs/notes/judge-normalization.md).

    바꾸려면 TEMPLATE_VERSION을 올리고 앵커를 다시 임베딩해야 한다.
    """
    assert normalization.render("RGB") == "RGB"


def test_template_version_is_pinned() -> None:
    """버전 문자열이 바뀌면 이 테스트가 깨진다.

    깨졌다면 앵커 재임베딩이 필요하다는 신호다. 값만 고쳐서 통과시키면 안 된다.
    """
    assert normalization.TEMPLATE_VERSION == "norm-v1"
