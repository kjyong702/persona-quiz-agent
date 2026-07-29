"""프롬프트 로더.

프롬프트를 코드 문자열이 아니라 버전 붙은 파일로 두는 이유가 두 가지다.
하나는 프롬프트를 고칠 때 코드 diff가 아니라 프롬프트 diff로 보이게 하려는 것이고,
다른 하나는 평가 도구(promptfoo)가 같은 파일을 그대로 읽어 회귀 테스트를
돌릴 수 있게 하려는 것이다. 코드에 박혀 있으면 둘 다 안 된다.
"""

from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"

JUDGE_PROMPT = "judge.v1"


@lru_cache(maxsize=None)
def load(name: str) -> str:
    return (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()
