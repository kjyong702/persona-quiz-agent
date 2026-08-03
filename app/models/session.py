from sqlalchemy import UniqueConstraint, Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base, TimestampMixin


class SessionStatus:
    IN_PROGRESS = "in_progress"
    FINISHED = "finished"


class JudgeMethod:
    """판정이 어느 경로로 났는지. Phase 3에서 기록한다."""

    EMBEDDING = "embedding"
    LLM = "llm"
    FALLBACK = "fallback"


class QuizSession(Base, TimestampMixin):
    __tablename__ = "quiz_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    quiz_set_id: Mapped[int] = mapped_column(ForeignKey("quiz_sets.id"))
    persona_id: Mapped[int] = mapped_column(ForeignKey("personas.id"))
    # 출제된 문제 번호. 0이면 아직 첫 문제를 받지 않은 상태
    current_order: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default=SessionStatus.IN_PROGRESS)


class SessionAnswer(Base, TimestampMixin):
    """답변 한 건과 그 판정 결과.

    is_correct만이 아니라 judge_method와 similarity를 함께 남긴다.
    임계값을 얼마로 둘지는 감이 아니라 이 데이터로 정하고(Phase 5),
    "정답률이 왜 이렇게 나왔나"를 사후에 경로별로 쪼개볼 수 있어야 한다.
    """

    __tablename__ = "session_answers"
    __table_args__ = (
        # **한 문항에 답변은 하나다.** 이 제약이 없으면 같은 답변이 동시에 두 번
        # 들어올 때 둘 다 저장된다. 판정에 수 초가 걸리므로 창이 넓다.
        #
        # 흔히 쓰는 멱등 키(idempotency key)를 따로 두지 않은 이유는
        # **자연 키가 이미 있기 때문**이다. 결제처럼 같은 금액을 두 번 보내는 것이
        # 정당할 수 있는 경우에는 클라이언트가 키를 만들어 보내야 하지만,
        # 퀴즈 한 문항에는 답변이 하나뿐이라 (session_id, question_id)가 유일하다.
        #
        # ⚠️ 되묻기(dev-plan Phase 9)를 넣으면 한 문항에 여러 답변이 오므로
        # 시도 번호를 제약에 넣어야 한다
        UniqueConstraint("session_id", "question_id", name="uq_session_question"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("quiz_sessions.id"), index=True)
    question_id: Mapped[int] = mapped_column(ForeignKey("questions.id"))
    answer_text: Mapped[str] = mapped_column(Text)
    is_correct: Mapped[bool] = mapped_column(Boolean)
    judge_method: Mapped[str] = mapped_column(String(20))
    # 임베딩을 못 쓴 폴백 판정이면 유사도가 없다
    similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 다른 문제의 기대 정답과의 최고 유사도.
    # similarity와의 차이가 margin이고, 그 하한을 정하는 근거가 이 값이다
    rival_similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 유사도 값을 나중에 해석하려면 어떤 모델로 어떤 문자열 형태를 재었는지가 있어야 한다.
    # 이 둘이 없으면 모델이나 템플릿을 바꿨을 때 과거 데이터를 쓸 수 있는지 판단할 수 없다
    embedding_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    template_version: Mapped[str | None] = mapped_column(String(20), nullable=True)
