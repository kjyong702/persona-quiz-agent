"""시드 로더.

    uv run python -m scripts.seed

seed/의 JSON을 읽어 DB에 넣는다. 기존 시드 데이터는 지우고 다시 넣는다.

Phase 3에서 기대 정답 임베딩을 ChromaDB에 적재하는 단계가 여기에 붙는다.
그때 임베딩 모델 ID와 렌더링 템플릿 버전을 함께 기록해야 하므로,
"시드 로드"는 앞으로 DB와 벡터 스토어 양쪽을 한 번에 맞추는 진입점이 된다.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

from app.core.database import Base, SessionLocal, engine
from app.repositories import seed_repository

SEED_DIR = Path(__file__).resolve().parent.parent / "seed"


def _load(filename: str) -> list[dict[str, Any]]:
    return json.loads((SEED_DIR / filename).read_text(encoding="utf-8"))


async def main() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    personas = _load("personas.json")
    quiz_sets = _load("quiz-sets.json")

    async with SessionLocal() as db:
        persona_count, set_count, question_count = await seed_repository.replace_all(
            db, personas, quiz_sets
        )

    print(
        f"시드 완료: 페르소나 {persona_count}개, "
        f"퀴즈 세트 {set_count}개, 문제 {question_count}개"
    )


if __name__ == "__main__":
    asyncio.run(main())
