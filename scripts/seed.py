"""시드 로더.

    uv run python -m scripts.seed

seed/의 JSON을 DB에 넣고, 기대 정답을 임베딩해 벡터 스토어에 적재한다.
기존 시드 데이터와 앵커는 지우고 다시 넣는다.

앵커 적재를 DB 시드와 한 명령에 묶은 이유: 둘이 어긋나면 판정이 조용히
망가진다. 문제는 있는데 앵커가 없으면 그 문제는 전부 LLM 폴백으로 넘어가고,
앵커만 남아 있으면 지워진 문제의 벡터가 rival_similarity를 오염시킨다.

OPENAI_API_KEY가 없으면 DB만 넣고 앵커 적재는 건너뛴다.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

from app.core import embedding, normalization, vector_store
from app.core.config import settings
from app.core.database import Base, SessionLocal, engine
from app.core.exceptions import ExternalServiceError
from app.repositories import quiz_repository, seed_repository

SEED_DIR = Path(__file__).resolve().parent.parent / "seed"


def _load(filename: str) -> list[dict[str, Any]]:
    return json.loads((SEED_DIR / filename).read_text(encoding="utf-8"))


async def _load_anchors() -> int:
    """기대 정답을 임베딩해 벡터 스토어에 적재하고 적재 건수를 돌려준다."""
    async with SessionLocal() as db:
        questions = await quiz_repository.list_all_questions(db)

    documents: list[str] = []
    pending: list[tuple[str, int, int, str]] = []  # id, question_id, quiz_set_id, 원문
    for question in questions:
        for index, raw_text in enumerate(question.expected_answers):
            documents.append(normalization.render(raw_text))
            pending.append(
                (f"q{question.id}-a{index}", question.id, question.quiz_set_id, raw_text)
            )

    # 한 번의 호출로 전부 임베딩한다. 건별로 부르면 호출 수만큼 쿼터를 쓴다
    vectors = await embedding.embed(documents)

    anchors = [
        vector_store.Anchor(
            id=anchor_id,
            embedding=vector,
            document=document,
            metadata={
                "question_id": question_id,
                "quiz_set_id": quiz_set_id,
                "raw_text": raw_text,
                "embedding_model": settings.embedding_model,
                "template_version": normalization.TEMPLATE_VERSION,
            },
        )
        for (anchor_id, question_id, quiz_set_id, raw_text), document, vector in zip(
            pending, documents, vectors, strict=True
        )
    ]
    await vector_store.replace_anchors(anchors)
    return len(anchors)


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
        f"DB 시드 완료: 페르소나 {persona_count}개, "
        f"퀴즈 세트 {set_count}개, 문제 {question_count}개"
    )

    if not settings.openai_api_key:
        print(
            "OPENAI_API_KEY가 없어 앵커 임베딩을 건너뜁니다. "
            "이 상태에서는 모든 답변이 LLM 폴백 경로로 판정됩니다."
        )
        return

    try:
        anchor_count = await _load_anchors()
    except ExternalServiceError as exc:
        print(f"앵커 적재 실패: {exc}")
        return

    print(
        f"앵커 적재 완료: {anchor_count}건 "
        f"(모델 {settings.embedding_model}, 템플릿 {normalization.TEMPLATE_VERSION})"
    )


if __name__ == "__main__":
    asyncio.run(main())
