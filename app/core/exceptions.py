class ErrorCode:
    """API 에러 코드. 문자열을 각 서비스에서 직접 만들지 않고 여기 모은다.

    코드가 흩어지면 같은 상황에 다른 코드가 나가고 클라이언트가 분기할 수 없다.
    """

    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    QUIZ_SET_NOT_FOUND = "QUIZ_SET_NOT_FOUND"
    PERSONA_NOT_FOUND = "PERSONA_NOT_FOUND"
    QUIZ_SET_EMPTY = "QUIZ_SET_EMPTY"
    SESSION_FINISHED = "SESSION_FINISHED"
    NO_ACTIVE_QUESTION = "NO_ACTIVE_QUESTION"


class AppError(Exception):
    """도메인 에러의 기반.

    서비스가 던지고 main의 예외 핸들러가 공통 응답 래퍼로 바꾼다.
    서비스가 HTTP 상태 코드를 직접 다루지 않게 하려는 것이다.
    """

    status_code = 400

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class NotFoundError(AppError):
    status_code = 404


class DomainError(AppError):
    status_code = 400
