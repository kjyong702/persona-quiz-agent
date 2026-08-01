"""페르소나 시스템 프롬프트 조립.

**프리픽스와 델타를 나누는 것이 이 모듈의 존재 이유다.**

    [ 공통 프리픽스 1,032토큰 ][ 페르소나 델타 ~110토큰 ]
      모든 세션이 공유             진행자마다 다름

앞 구간이 **바이트 단위로 같아야** 제공사 프롬프트 캐싱이 걸린다. 그래서
프리픽스에는 세션 ID, 시각, 문제 번호 같은 **매번 달라지는 값을 절대 넣지 않는다.**
하나라도 섞이면 프리픽스가 매 요청 달라져 캐싱이 영영 안 걸린다.

측정으로 확인한 조건은 `docs/notes/prompt-caching.md`에 있다. 요약하면
최소 1,024토큰에 128토큰 단위이고, 이 프리픽스는 1,032토큰이라 아슬아슬하게
대상이 된다. **프롬프트를 줄이면 캐싱이 꺼진다는 뜻이기도 하다.**

델타를 뒤에 두는 이유는 캐싱 때문만이 아니다. 고정 규칙과 변동 설정이 섞여 있으면
"말투를 바꿨더니 정답 유출 금지가 흔들렸다" 같은 일이 생겨도 원인을 못 찾는다.
"""

from app.core import prompts
from app.models import Persona

BASE_PROMPT = "persona-base.v1"

# 진행 상황. 프롬프트의 "상황별로 할 일" 절과 이름이 같아야 한다
OPENING = "opening"
QUESTION = "question"
CORRECT = "correct"
INCORRECT = "incorrect"
CLOSING = "closing"


def build_system_prompt(persona: Persona) -> str:
    """공통 프리픽스 + 페르소나 델타.

    **프리픽스가 먼저 오고 델타가 뒤에 온다.** 순서를 바꾸면 캐싱이 걸리지 않는다.
    """
    return (
        f"{prompts.load(BASE_PROMPT)}\n\n"
        "# 이번 진행자\n\n"
        f"이름: {persona.name}\n"
        f"성격: {persona.personality}\n"
        f"말투: {persona.speech_style}\n"
        f"리액션: {persona.reaction_style}"
    )


def build_user_message(
    situation: str,
    *,
    question_text: str | None = None,
    order_no: int | None = None,
    total: int | None = None,
    score: int | None = None,
) -> str:
    """상황별 사용자 메시지.

    변동 값은 **전부 여기 있다.** 시스템 프롬프트 쪽에는 하나도 넣지 않는다.
    """
    lines = [f"상황: {situation}"]
    if order_no is not None and total is not None:
        lines.append(f"진행: {total}문제 중 {order_no}번째")
    if question_text is not None:
        lines.append(f"문제: {question_text}")
    if score is not None and total is not None:
        lines.append(f"최종 성적: {total}문제 중 {score}문제 정답")
    return "\n".join(lines)
