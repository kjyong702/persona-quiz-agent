from typing import Any

from fastapi import APIRouter

from app.core import metrics
from app.schemas.common import ApiResponse, ok

router = APIRouter(tags=["ops"])


@router.get("/metrics", response_model=ApiResponse[dict[str, Any]])
async def read_metrics() -> ApiResponse[dict[str, Any]]:
    return ok(metrics.snapshot())
