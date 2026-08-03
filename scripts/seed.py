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
import sys
from pathlib import Path
from typing import Any

from app.core import embedding, normalization, vector_store
from app.core.config import settings
from app.core.database import Base, SessionLocal, engine
from app.core.exceptions import ExternalServiceError
from app.repositories import quiz_repository, seed_repository

SEED_DIR = Path(__file__).resolve().parent.parent / "seed"


class SeedIntegrityError(RuntimeError):
    """시드 결과가 판정에 쓸 수 없는 상태임을 뜻한다.

    외부 API 실패(ExternalServiceError)와 구분한다. 저쪽은 다시 돌리면 되는
    일시적 실패이고, 이쪽은 데이터가 어긋나 있어 다시 돌려도 같은 결과가 나온다.
    """


def _load(filename: str) -> list[dict[str, Any]]:
    return json.loads((SEED_DIR / filename).read_text(encoding="utf-8"))


async def _load_anchors() -> int:
    """기대 정답을 임베딩해 벡터 스토어에 적재하고 적재 건수를 돌려준다."""
    async with SessionLocal() as db:
        questions = await quiz_repository.list_all_questions(db)

    if not questions:
        raise SeedIntegrityError("문항이 하나도 없습니다. DB 시드가 먼저 끝나야 합니다")

    # 기대 정답이 비어 있는 문항은 앵커가 하나도 안 생긴다. 그런 문항은 판정에서
    # similarity=None이 되어 전부 LLM 폴백으로 가는데, 예외도 로그도 남지 않는다.
    # 여기서 끊지 않으면 "시드는 성공했는데 그 문항만 하이브리드가 꺼진" 상태가 된다
    without_answers = [q.id for q in questions if not q.expected_answers]
    if without_answers:
        raise SeedIntegrityError(f"기대 정답이 없는 문항이 있습니다: {without_answers}")

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
                # embedding_model과 template_version은 vector_store가 찍는다.
                # 여기서 손으로 넣으면 호출자마다 빠뜨릴 수 있다
            },
        )
        for (anchor_id, question_id, quiz_set_id, raw_text), document, vector in zip(
            pending, documents, vectors, strict=True
        )
    ]
    await vector_store.replace_anchors(anchors)

    # 적재했다고 믿지 않고 스토어에 물어본다. 개수가 어긋나도 조회는 정상 동작해서
    # 판정 품질만 조용히 떨어지므로, 여기가 그걸 잡을 수 있는 유일한 지점이다
    stored = await vector_store.count()
    if stored != len(anchors):
        raise SeedIntegrityError(
            f"앵커 적재 수가 맞지 않습니다: 넣으려 한 {len(anchors)}건, 스토어 {stored}건"
        )
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
        # 키가 없는 것은 의도된 사용법이다(DB만 채우고 싶을 때). 실패가 아니므로
        # 종료 코드는 0이지만, 이 상태가 무엇을 뜻하는지는 분명히 알린다
        print(
            "OPENAI_API_KEY가 없어 앵커 임베딩을 건너뜁니다. "
            "이 상태에서는 모든 답변이 LLM 폴백 경로로 판정됩니다."
        )
        return

    anchor_count = await _load_anchors()
    print(
        f"앵커 적재 완료: {anchor_count}건 "
        f"(모델 {settings.embedding_model}, 템플릿 {normalization.TEMPLATE_VERSION})"
    )


if __name__ == "__main__":
    # 앵커 적재 실패를 종료 코드로 알린다. 예전에는 메시지만 찍고 0으로 끝나서
    # "DB는 채워졌는데 앵커는 없는" 상태가 성공처럼 보였다. 그 상태로 서버를 띄우면
    # 모든 판정이 LLM 폴백으로 가면서 비용은 오르고 하이브리드는 꺼져 있는데,
    # 어디에도 실패 신호가 없어 알아챌 방법이 없었다
    try:
        asyncio.run(main())
    except (ExternalServiceError, SeedIntegrityError) as exc:
        print(f"시드 실패: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
