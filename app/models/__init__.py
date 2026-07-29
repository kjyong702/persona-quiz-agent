"""SQLAlchemy 모델. docs/db-schema.md와 1대1로 맞춘다.

관계(relationship)를 두지 않고 필요한 조회를 리포지토리에서 명시적으로 쓴다.
async 세션에서 관계를 지연 로딩하면 접근 시점에 예외가 나거나 의도치 않은
추가 쿼리가 나가기 때문에, 무엇이 언제 조회되는지를 코드에 드러내는 쪽을 택했다.
"""

from app.models.persona import Persona
from app.models.quiz import Question, QuizSet
from app.models.session import JudgeMethod, QuizSession, SessionAnswer, SessionStatus

__all__ = [
    "JudgeMethod",
    "Persona",
    "Question",
    "QuizSession",
    "QuizSet",
    "SessionAnswer",
    "SessionStatus",
]
