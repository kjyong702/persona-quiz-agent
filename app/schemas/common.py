from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ErrorBody(BaseModel):
    code: str
    message: str


class ApiResponse(BaseModel, Generic[T]):
    """공통 응답 래퍼. 성공이든 실패든 껍데기 모양이 같다.

    제네릭으로 둔 덕에 라우터에 response_model=ApiResponse[list[...]]를 걸면
    OpenAPI 문서에도 실제 payload 타입이 그대로 나온다.
    """

    data: T | None = None
    error: ErrorBody | None = None


def ok(data: T) -> ApiResponse[T]:
    return ApiResponse(data=data)
