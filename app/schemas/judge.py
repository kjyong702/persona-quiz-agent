from pydantic import BaseModel


class JudgeResult(BaseModel):
    """판정 한 건의 결과와 근거.

    응답에 그대로 나가는 모델이 아니다. 판정 서비스가 만들어 세션 서비스와
    리포지토리에 넘기는 공용 어휘이고, 클라이언트에게는 이 중 일부만 나간다.
    """

    is_correct: bool
    judge_method: str
    similarity: float | None = None
    rival_similarity: float | None = None
    embedding_model: str | None = None
    template_version: str | None = None
