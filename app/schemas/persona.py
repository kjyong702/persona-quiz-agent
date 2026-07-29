from pydantic import BaseModel


class PersonaSummary(BaseModel):
    """페르소나 목록 항목.

    reaction_style은 내보내지 않는다. 프롬프트 조립에만 쓰는 내부 값이고
    호스트가 어떤 기준으로 반응하는지를 미리 노출할 이유가 없다.
    """

    id: int
    name: str
    personality: str
    speech_style: str
