"""라우터 계층 테스트.

서비스 로직은 단위 테스트에서 다뤘으므로 여기서는 껍데기만 본다.
성공이든 실패든 응답이 공통 래퍼 모양을 유지하는지가 관심사다.
"""

from types import SimpleNamespace

from httpx import AsyncClient


async def test_personas_endpoint_wraps_data(
    client: AsyncClient, seeded: SimpleNamespace
) -> None:
    response = await client.get("/personas")

    assert response.status_code == 200
    body = response.json()
    assert body["error"] is None
    assert body["data"][0]["name"] == "불꽃"


async def test_quiz_sets_endpoint_wraps_data(
    client: AsyncClient, seeded: SimpleNamespace
) -> None:
    response = await client.get("/quiz-sets")

    assert response.status_code == 200
    body = response.json()
    assert body["error"] is None
    assert {item["title"] for item in body["data"]} == {"테스트 세트", "빈 세트"}


async def test_start_session_returns_201(
    client: AsyncClient, seeded: SimpleNamespace
) -> None:
    response = await client.post(
        "/sessions",
        json={"quiz_set_id": seeded.quiz_set_id, "persona_id": seeded.persona_id},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["error"] is None
    assert body["data"]["session_id"] > 0


async def test_next_question_endpoint(
    client: AsyncClient, seeded: SimpleNamespace
) -> None:
    created = await client.post(
        "/sessions",
        json={"quiz_set_id": seeded.quiz_set_id, "persona_id": seeded.persona_id},
    )
    session_id = created.json()["data"]["session_id"]

    response = await client.post(f"/sessions/{session_id}/next")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["finished"] is False
    assert data["order_no"] == 1


async def test_domain_error_uses_error_wrapper(client: AsyncClient) -> None:
    response = await client.get("/sessions/9999")

    assert response.status_code == 404
    body = response.json()
    assert body["data"] is None
    assert body["error"]["code"] == "SESSION_NOT_FOUND"


async def test_request_validation_returns_422(client: AsyncClient) -> None:
    """스키마 검증 실패는 FastAPI 기본 422를 그대로 쓴다 (CLAUDE.md API 규칙)."""
    response = await client.post(
        "/sessions", json={"quiz_set_id": 0, "persona_id": 1}
    )

    assert response.status_code == 422
