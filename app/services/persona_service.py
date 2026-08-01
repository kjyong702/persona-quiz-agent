from sqlalchemy.ext.asyncio import AsyncSession

from app.core import llm, persona_prompt
from app.core.exceptions import ExternalServiceError
from app.models import Persona
from app.repositories import persona_repository
from app.schemas.persona import PersonaSummary


async def list_personas(db: AsyncSession) -> list[PersonaSummary]:
    personas = await persona_repository.list_all(db)
    return [
        PersonaSummary(
            id=p.id,
            name=p.name,
            personality=p.personality,
            speech_style=p.speech_style,
        )
        for p in personas
    ]


# --- 진행 멘트 생성 (Phase 4) -------------------------------------------------
#
# 판정(judge_service)과 이 아래는 **같은 모델을 쓰지만 요구가 반대**다.
#
#   판정      같은 입력에 같은 출력이 나와야 한다   temperature 0
#   진행 멘트  매번 조금씩 달라야 자연스럽다        temperature 0.8
#
# 실패 처리도 반대다. 판정은 틀리면 데이터가 오염되므로 차라리 503으로 멈추지만,
# **멘트는 없어도 퀴즈가 진행된다.** 진행자가 말을 못 한다고 출제까지 막으면
# 그게 더 큰 실패다. 그래서 여기서는 예외를 밖으로 던지지 않고 None을 돌려준다.


async def opening_message(persona: Persona, *, total: int) -> str | None:
    return await _generate(
        persona, persona_prompt.build_user_message(persona_prompt.OPENING, total=total)
    )


async def question_message(
    persona: Persona, *, question_text: str, order_no: int, total: int
) -> str | None:
    return await _generate(
        persona,
        persona_prompt.build_user_message(
            persona_prompt.QUESTION,
            question_text=question_text,
            order_no=order_no,
            total=total,
        ),
    )


async def reaction_message(
    persona: Persona, *, is_correct: bool, order_no: int, total: int
) -> str | None:
    """채점 결과에 대한 반응.

    **문제 문장도 기대 정답도 넘기지 않는다.** 진행자가 알 필요가 없고,
    프롬프트에 없으면 흘릴 수도 없다. 정답 유출을 지시문으로만 막지 않고
    **입력에서 아예 빼는 것**이 확실하다.
    """
    situation = persona_prompt.CORRECT if is_correct else persona_prompt.INCORRECT
    return await _generate(
        persona,
        persona_prompt.build_user_message(situation, order_no=order_no, total=total),
    )


async def closing_message(persona: Persona, *, score: int, total: int) -> str | None:
    return await _generate(
        persona,
        persona_prompt.build_user_message(
            persona_prompt.CLOSING, score=score, total=total
        ),
    )


async def _generate(persona: Persona, user_message: str) -> str | None:
    try:
        return await llm.generate_host_message(
            persona_prompt.build_system_prompt(persona), user_message
        )
    except ExternalServiceError:
        # 진행자가 말을 못 해도 퀴즈는 굴러가야 한다.
        # host_message가 null이면 클라이언트는 멘트 없이 문제만 보여주면 된다
        return None
