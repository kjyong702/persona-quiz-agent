from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Persona


async def list_all(db: AsyncSession) -> Sequence[Persona]:
    result = await db.execute(select(Persona).order_by(Persona.id))
    return result.scalars().all()


async def get(db: AsyncSession, persona_id: int) -> Persona | None:
    return await db.get(Persona, persona_id)
