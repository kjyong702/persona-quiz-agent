"""유사도 분포 측정. 평가 데이터셋 전건을 실제 판정 경로에 태워 원시 수치를 남긴다.

    uv run python -m scripts.measure

**측정과 분석을 나눈 이유가 이 스크립트의 설계다.**

임계값을 정하려면 같은 데이터에 임계값만 바꿔가며 수십 번 계산해봐야 한다.
그때마다 임베딩을 다시 부르면 돈과 시간이 그만큼 들고, 더 나쁜 것은 **호출할 때마다
값이 미세하게 달라질 수 있어 비교 대상이 흔들린다**는 점이다.

그래서 여기서는 유사도까지만 재서 `eval/measurements.json`에 굳히고,
임계값 판단은 `scripts/analyze.py`가 그 파일만 읽어 한다. 분석은 공짜고 몇 번을
돌려도 같은 입력이다.

임베딩은 캐시한다. 같은 (답변, 모델, 템플릿 버전) 조합은 다시 부르지 않는다.
셋 중 하나라도 바뀌면 그 항목은 다시 부른다. 다른 조건에서 만든 벡터를 섞으면
비교가 성립하지 않기 때문이다.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from app.core import embedding, normalization, vector_store
from app.core.config import settings
from app.core.exceptions import ExternalServiceError

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "eval" / "dataset.json"
CACHE = ROOT / "eval" / ".embedding-cache.json"
OUT = ROOT / "eval" / "measurements.json"

# 한 번에 보내는 건수. 제공사 상한(배열 2048건)보다 훨씬 낮게 잡는다.
# 크게 잡을수록 왕복은 줄지만 한 번 실패했을 때 다시 부르는 양이 커진다
CHUNK = 100


def _cache_key(answer: str) -> str:
    """모델과 템플릿 버전을 키에 넣는다. 둘 중 하나가 바뀌면 캐시가 자동으로 빗나간다."""
    return f"{settings.embedding_model}|{normalization.TEMPLATE_VERSION}|{answer}"


def _load_cache() -> dict[str, list[float]]:
    if not CACHE.exists():
        return {}
    try:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print("임베딩 캐시가 깨져 있어 무시하고 새로 만듭니다", file=sys.stderr)
        return {}


async def _embed_all(answers: list[str]) -> dict[str, list[float]]:
    """캐시에 없는 것만 임베딩한다."""
    cache = _load_cache()
    # 정규화 후 빈 문자열이 되는 답변은 임베딩에 보내지 않는다. 제공사가 400을 준다.
    # 판정 경로도 같은 이유로 이런 입력을 임베딩 전에 걷어낸다(judge_service 참고)
    todo = [
        a
        for a in dict.fromkeys(answers)
        if _cache_key(a) not in cache and normalization.render(a)
    ]

    if todo:
        print(f"임베딩 {len(todo)}건 (캐시 적중 {len(set(answers)) - len(todo)}건)")
        for i in range(0, len(todo), CHUNK):
            batch = todo[i : i + CHUNK]
            # 판정 경로와 **같은 정규화**를 태운다. 여기서 다르면 측정값이 실제
            # 서비스와 다른 것을 재게 된다
            vectors = await embedding.embed([normalization.render(a) for a in batch])
            for answer, vector in zip(batch, vectors, strict=True):
                cache[_cache_key(answer)] = vector
            print(f"  {min(i + CHUNK, len(todo))}/{len(todo)}")
            CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    else:
        print("전부 캐시 적중. 임베딩 호출 없음")

    return cache


async def main() -> None:
    if not DATASET.exists():
        raise SystemExit(f"데이터셋이 없습니다: {DATASET}")

    rows: list[dict[str, Any]] = json.loads(DATASET.read_text(encoding="utf-8"))
    print(f"데이터셋 {len(rows)}건")

    stored = await vector_store.count()
    if stored == 0:
        raise SystemExit(
            "벡터 스토어에 앵커가 없습니다. 먼저 `uv run python -m scripts.seed`를 돌리세요"
        )
    print(f"앵커 {stored}건")

    cache = await _embed_all([r["answer"] for r in rows])

    measured = []
    empty_rendered = 0
    for row in rows:
        rendered = normalization.render(row["answer"])
        if not rendered:
            # 임베딩 경로에 진입조차 못 하는 항목. 실서비스에서도 폴백으로 간다.
            # 유사도 통계에서 빼되 몇 건인지는 남긴다
            empty_rendered += 1
            measured.append(
                {
                    **row,
                    "similarity": None,
                    "rival_similarity": None,
                    "margin": None,
                    "rendered": "",
                    "rendered_empty": True,
                    "embedding_model": settings.embedding_model,
                    "template_version": normalization.TEMPLATE_VERSION,
                }
            )
            continue
        vector = cache[_cache_key(row["answer"])]
        # 판정과 **같은 함수**를 부른다. 여기서 직접 코사인을 계산하면
        # 실제 서비스가 겪는 것(메타데이터 필터, ANN 근사)을 재지 못한다
        match = await vector_store.match(vector, row["question_id"])
        sim = match.similarity
        rival = match.rival_similarity
        measured.append(
            {
                **row,
                "similarity": sim,
                "rival_similarity": rival,
                "margin": None if (sim is None or rival is None) else sim - rival,
                "rendered": rendered,
                "rendered_empty": False,
                "embedding_model": settings.embedding_model,
                "template_version": normalization.TEMPLATE_VERSION,
            }
        )

    # 정규화로 비워진 것이 아닌데 유사도가 없다면 앵커가 없다는 뜻이다.
    # 시드가 어긋난 상태이므로 결과를 믿으면 안 된다
    missing = [
        m["id"] for m in measured if m["similarity"] is None and not m["rendered_empty"]
    ]
    if missing:
        raise SystemExit(f"앵커를 못 찾은 항목이 있습니다(시드 확인 필요): {missing[:10]}")
    if empty_rendered:
        print(f"정규화 후 빈 문자열 {empty_rendered}건은 임베딩 경로 진입 불가로 표시했습니다")

    OUT.write_text(json.dumps(measured, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n측정 완료 -> {OUT.relative_to(ROOT)} ({len(measured)}건)")
    print("다음: uv run python -m scripts.analyze")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except ExternalServiceError as exc:
        print(f"측정 실패: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
