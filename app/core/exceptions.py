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
    JUDGE_UNAVAILABLE = "JUDGE_UNAVAILABLE"
    INDEX_DRIFT = "INDEX_DRIFT"
    REQUEST_TIMEOUT = "REQUEST_TIMEOUT"


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


class ServiceUnavailableError(AppError):
    status_code = 503


class ExternalServiceError(Exception):
    """외부 의존 호출 실패.

    AppError와 계통을 나눈 이유: 이건 클라이언트에게 그대로 나가는 에러가 아니라
    서비스가 붙잡아서 다른 경로로 갈지(폴백) 포기할지(503) 판단할 재료다.
    """


class EmbeddingUnavailableError(ExternalServiceError):
    pass


class VectorStoreUnavailableError(ExternalServiceError):
    pass


class LLMUnavailableError(ExternalServiceError):
    pass


class IndexDriftError(ExternalServiceError):
    """인덱스가 지금 설정과 다른 모델이나 템플릿으로 만들어졌다.

    **다른 ExternalServiceError와 성격이 다르다.** 나머지는 일시적 장애라
    재시도하거나 다른 경로로 갈 수 있지만, 이건 재시도해도 낫지 않는 구성 오류다.
    사람이 `scripts.seed`를 다시 돌려야 풀린다.

    폴백하지 않고 멈추는 이유는 조용히 틀리는 쪽이 더 나쁘기 때문이다.
    임베딩 공간이 다르면 유사도는 계산되지만 의미가 없다. 값이 나오므로
    아무도 알아채지 못한 채 임계값 판정이 계속 틀린다.
    """
