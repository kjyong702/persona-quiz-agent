from pydantic import BaseModel


class QuizSetSummary(BaseModel):
    id: int
    title: str
    description: str
    question_count: int
