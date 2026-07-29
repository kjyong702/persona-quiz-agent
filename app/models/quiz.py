from sqlalchemy import JSON, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base, TimestampMixin


class QuizSet(Base, TimestampMixin):
    __tablename__ = "quiz_sets"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(Text)


class Question(Base):
    """세트 안의 문제 하나.

    expected_answers는 "대표 정답 + 허용 변형" 목록이다. 별도 테이블로 빼지 않은 이유는
    이 목록을 단독으로 조회하거나 검색할 일이 없고 항상 문제와 함께 읽히기 때문이다.

    이 목록이 Phase 3 판정의 고정 축(앵커)이 된다. 사용자 답변을 다른 사용자 답변과
    비교하지 않고 항상 이 앵커와 비교하기 때문에 판정 기준이 흔들리지 않는다.
    """

    __tablename__ = "questions"
    __table_args__ = (UniqueConstraint("quiz_set_id", "order_no"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    quiz_set_id: Mapped[int] = mapped_column(ForeignKey("quiz_sets.id"), index=True)
    order_no: Mapped[int] = mapped_column(Integer)
    question_text: Mapped[str] = mapped_column(Text)
    expected_answers: Mapped[list[str]] = mapped_column(JSON)
