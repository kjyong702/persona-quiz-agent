"""임베딩 입력 정규화.

기대 정답과 사용자 답변을 항상 같은 형태로 만들어야 유사도를 비교할 수 있다.
무엇을 하고 무엇을 하지 않는지, 왜 그렇게 정했는지는
docs/notes/judge-normalization.md에 적어두었다.

**이 함수를 고치면 TEMPLATE_VERSION을 올리고 앵커를 전부 다시 임베딩해야 한다.**
버전이 다른 벡터가 섞이면 유사도 비교가 성립하지 않는다.
"""

import re
import unicodedata

TEMPLATE_VERSION = "norm-v1"

_WHITESPACE = re.compile(r"\s+")
_TRAILING_PUNCTUATION = re.compile(r"[\s.,!?;:~·…。！？]+$")


def render(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _WHITESPACE.sub(" ", normalized).strip()
    return _TRAILING_PUNCTUATION.sub("", normalized)
