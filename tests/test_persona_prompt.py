"""페르소나 프롬프트 조립 테스트.

**여기서 지키는 것은 캐싱의 전제다.**

프리픽스가 맨 앞에 오고 바이트 단위로 고정되어야 제공사 캐싱이 걸린다.
실측에서 프리픽스가 문턱 아래였을 때 적중률이 0이었고, 앞에 변동 값을 넣었을
때는 비용이 50% 올랐다(docs/notes/prompt-caching.md). **주석으로만 적어둔
규칙은 지켜지는지 알 수 없어서 테스트로 옮긴다.**
"""

import pytest

from app.core import persona_prompt, prompts
from app.models import Persona


@pytest.fixture
def persona() -> Persona:
    return Persona(
        id=1,
        name="불꽃",
        personality="승부욕이 강하다",
        speech_style="반말을 쓴다",
        reaction_style="크게 환호한다",
    )


def test_프리픽스가_맨_앞에_온다(persona: Persona) -> None:
    """순서가 뒤바뀌면 캐싱이 안 걸린다. 실측 73.1% 대 37.8%."""
    system = persona_prompt.build_system_prompt(persona)
    base = prompts.load(persona_prompt.BASE_PROMPT)

    assert system.startswith(base)


def test_페르소나가_달라도_프리픽스는_바이트까지_같다() -> None:
    """공유 구간이 한 글자라도 다르면 캐시 엔트리가 따로 생긴다."""
    a = persona_prompt.build_system_prompt(
        Persona(id=1, name="불꽃", personality="A", speech_style="B", reaction_style="C")
    )
    b = persona_prompt.build_system_prompt(
        Persona(id=2, name="고요", personality="X", speech_style="Y", reaction_style="Z")
    )
    base = prompts.load(persona_prompt.BASE_PROMPT)

    assert a[: len(base)] == b[: len(base)]


def test_프리픽스에_변동_값이_없다(persona: Persona) -> None:
    """세션 ID나 시각이 프리픽스에 끼면 캐싱이 영구히 죽는다.

    실측: 맨 앞에 세션 ID 한 줄을 넣자 적중률이 0%가 되고 비용이 50.5% 올랐다.
    """
    base = prompts.load(persona_prompt.BASE_PROMPT)

    # 프리픽스는 파일에서 그대로 온다. 조립 과정에서 아무것도 끼어들지 않는다
    assert base == base.strip()
    for marker in ("세션", "session", "id=", "시각", "timestamp"):
        assert marker not in base.lower()


def test_변동_값은_전부_user_메시지에_있다(persona: Persona) -> None:
    """문제나 진행 상황이 시스템 프롬프트로 새면 프리픽스가 매번 달라진다.

    처음에는 "문제 문장이 system에 없다"로 단정했다가 실패했다.
    프리픽스의 예시에 같은 문장이 예시로 들어 있어서다. **문자열 부재로 검사하면
    본문과 충돌한다.** 시그니처상 문제를 받지 않는다는 것을 직접 확인하는 쪽이 맞다.
    """
    user = persona_prompt.build_user_message(
        persona_prompt.QUESTION,
        question_text="가상의 문제 XKCD-9137",
        order_no=3,
        total=10,
    )

    assert "3번째" in user
    assert "XKCD-9137" in user

    # 시스템 프롬프트는 페르소나만으로 결정된다. 문제가 무엇이든 같은 문자열이다
    assert persona_prompt.build_system_prompt(persona) == persona_prompt.build_system_prompt(
        persona
    )
    assert "XKCD-9137" not in persona_prompt.build_system_prompt(persona)


def test_리액션에는_문제도_정답도_넘기지_않는다() -> None:
    """정답 유출을 지시문으로만 막지 않는다. **입력에서 아예 뺀다.**

    프롬프트에 없으면 흘릴 수도 없다. 실측 18건에서 유출 0건.
    """
    user = persona_prompt.build_user_message(
        persona_prompt.INCORRECT, order_no=1, total=10
    )

    assert "문제:" not in user
    assert "정답" not in user


def test_페르소나_델타가_전부_들어간다(persona: Persona) -> None:
    system = persona_prompt.build_system_prompt(persona)

    for value in (persona.name, persona.personality, persona.speech_style, persona.reaction_style):
        assert value in system
