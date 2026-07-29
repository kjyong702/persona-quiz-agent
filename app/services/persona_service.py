from sqlalchemy.ext.asyncio import AsyncSession

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
