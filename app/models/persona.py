from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base, TimestampMixin


class Persona(Base, TimestampMixin):
    """퀴즈를 진행하는 AI 호스트의 프로필.

    이 세 필드(성격, 말투, 리액션 성향)가 Phase 4에서 시스템 프롬프트로 조립된다.
    프롬프트 원문을 컬럼에 넣지 않는 이유는, 프롬프트를 바꿀 때 DB 데이터가 아니라
    코드(템플릿)만 고치면 되게 하려는 것이다.
    """

    __tablename__ = "personas"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50))
    personality: Mapped[str] = mapped_column(Text)
    speech_style: Mapped[str] = mapped_column(Text)
    reaction_style: Mapped[str] = mapped_column(Text)
