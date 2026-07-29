from typing import Any

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    Persona,
    Question,
    QuizSession,
    QuizSet,
    SessionAnswer,
)


async def replace_all(
    db: AsyncSession,
    personas: list[dict[str, Any]],
    quiz_sets: list[dict[str, Any]],
) -> tuple[int, int, int]:
    """시드 데이터를 통째로 갈아끼운다. 개발용이다.

    문제 순서(order_no)를 JSON에 적지 않고 배열 순서에서 만든다.
    사람이 시드 파일을 편집할 때 번호를 손으로 맞추다 빠뜨리는 일을 막는다.

    반환값은 (페르소나 수, 세트 수, 문제 수).
    """
    # FK 참조 순서의 역순으로 지운다
    for model in (SessionAnswer, QuizSession, Question, QuizSet, Persona):
        await db.execute(delete(model))

    db.add_all([Persona(**p) for p in personas])

    question_count = 0
    for raw_set in quiz_sets:
        quiz_set = QuizSet(title=raw_set["title"], description=raw_set["description"])
        db.add(quiz_set)
        await db.flush()  # quiz_set.id를 받아야 문제를 붙일 수 있다

        for order_no, raw_question in enumerate(raw_set["questions"], start=1):
            db.add(
                Question(
                    quiz_set_id=quiz_set.id,
                    order_no=order_no,
                    question_text=raw_question["question_text"],
                    expected_answers=raw_question["expected_answers"],
                )
            )
            question_count += 1

    await db.commit()
    return len(personas), len(quiz_sets), question_count
