from pydantic import BaseModel, Field


class SessionCreateRequest(BaseModel):
    quiz_set_id: int = Field(gt=0)
    persona_id: int = Field(gt=0)


class SessionCreated(BaseModel):
    session_id: int
    # 오프닝 멘트는 Phase 4 페르소나 레이어에서 채운다. 그전까지는 null
    host_message: str | None = None


class SessionState(BaseModel):
    status: str
    current_order: int
    total_questions: int
    correct_count: int


class AnswerRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=500)


class AnswerResult(BaseModel):
    """판정 결과.

    임베딩 모델 ID와 템플릿 버전, rival_similarity는 여기 없다.
    판정을 재현할 때 필요한 값이지 클라이언트가 알아야 할 것은 아니라서
    DB에만 남긴다.
    """

    is_correct: bool
    judge_method: str
    similarity: float | None = None
    # 리액션 멘트는 Phase 4 페르소나 레이어에서 채운다
    host_message: str | None = None


class NextQuestion(BaseModel):
    """다음 문제 응답.

    출제와 종료를 한 스키마로 둔다. 종료일 때 문제 필드는 전부 null이고
    finished 플래그로 구분한다. 클라이언트가 응답 모양이 아니라 필드 하나만
    보고 분기할 수 있게 하려는 것이다.
    """

    finished: bool = False
    question_id: int | None = None
    order_no: int | None = None
    question_text: str | None = None
    host_message: str | None = None
