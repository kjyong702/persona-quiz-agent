"""promptfoo가 진행 멘트 프롬프트를 만들 때 부르는 함수.

`judge_prompt.py`와 같은 원칙이다. **프롬프트를 여기 베끼지 않고 운영 코드를 부른다.**
페르소나 프롬프트는 공통 프리픽스와 페르소나 델타로 나뉘어 있고 순서까지
캐싱에 걸려 있어서, 베끼면 어느 쪽이 어긋났는지도 모르게 된다.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.core import persona_prompt  # noqa: E402


def host(context: dict) -> list[dict]:
    v = context["vars"]
    persona = SimpleNamespace(
        name=v["persona_name"],
        personality=v["personality"],
        speech_style=v["speech_style"],
        reaction_style=v["reaction_style"],
    )
    kwargs = {
        key: v[key]
        for key in ("question_text", "order_no", "total", "score")
        if v.get(key) not in (None, "")
    }
    return [
        {"role": "system", "content": persona_prompt.build_system_prompt(persona)},
        {"role": "user", "content": persona_prompt.build_user_message(v["situation"], **kwargs)},
    ]
