"""구조화 로깅.

**이름을 `logging.py`로 안 지은 이유**: 표준 라이브러리 `logging`과 헷갈린다.
같은 파일 안에서 둘 다 쓰기 때문에 이름이 겹치면 읽는 사람이 매번 멈춘다.

## 왜 필요한가

지금까지 **예외를 폴백으로 흡수하면 흔적이 남지 않았다.** 임베딩이 죽어도
퀴즈는 LLM으로 계속 돈다. 그건 옳은 동작인데 **밖에서 보면 아무 일도 없다.**
사후에 "왜 그날 LLM 비용이 두 배였지"를 물으면 답할 근거가 없다.

## 로깅과 트레이싱

**로깅은 일어난 일을 그때그때 적고, 트레이싱은 요청 문맥을 단계들에 걸쳐 보존한다.**
답변 하나를 판정하는 데 임베딩 1회 + LLM 0~1회가 나가므로, 그것들을 잇지 않으면
"이 판정이 왜 이렇게 나왔나"를 되짚을 수 없다.

여기서는 **요청 ID를 contextvar에 심어 모든 로그에 자동으로 붙이는** 정도만 한다.
OpenTelemetry 같은 분산 추적은 붙이지 않았다. 서비스가 하나뿐이라 경계를 넘는
전파가 필요 없고, 도구를 붙이는 것 자체가 이 프로젝트의 주제가 아니다.

## 무엇을 남기는가

`judge.completed` 한 줄에 **판정 하나의 전체 이야기**가 들어간다.

    request_id, question_id, method, similarity, is_correct,
    embedding_model, template_version, judge_prompt, judge_model, duration_ms

**프롬프트 버전을 남기는 것이 핵심이다.** 프롬프트는 코드와 다른 주기로 바뀌므로
커밋 해시만으로는 어느 버전이 그 판정을 냈는지 알 수 없다. `judge.v1`이 낸 판정과
`judge.v3`이 낸 판정이 로그에서 갈려야 프롬프트를 바꾼 효과를 사후에 볼 수 있다.

**답변 원문과 프롬프트 전문은 남기지 않는다.** 사용자 입력이고, 로그는 보통
평문으로 여러 곳에 복제된다. 어느 경로로 갔고 유사도가 얼마였는지만 있으면
비용과 품질을 추적하는 데는 충분하다.
"""

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

import structlog

_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def new_request_id() -> str:
    """요청마다 새로 만들어 이 문맥에 심는다. 이후 모든 로그에 자동으로 붙는다."""
    value = uuid.uuid4().hex[:12]
    _request_id.set(value)
    return value


def set_request_id(value: str) -> None:
    """앞단이 이미 ID를 붙여 보냈을 때 그걸 이어받는다."""
    _request_id.set(value)


def current_request_id() -> str:
    return _request_id.get()


def _add_request_id(_logger: Any, _name: str, event_dict: dict) -> dict:
    event_dict["request_id"] = _request_id.get()
    return event_dict


def configure(json_output: bool = True, level: str = "INFO") -> None:
    """부팅 시 한 번 부른다.

    `json_output`이 False면 사람이 읽는 형태로 나온다. 개발할 때만 쓴다.
    프로덕션에서 사람이 읽는 형태로 두면 수집기가 파싱을 못 한다.
    """
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _add_request_id,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            # 예외가 있으면 트레이스백을 구조화해 넣는다. 폴백으로 흡수한 예외를
            # 남기는 것이 이 모듈의 첫 번째 목적이라 이 프로세서가 핵심이다
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(ensure_ascii=False)
            if json_output
            else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level]
        ),
        cache_logger_on_first_use=True,
    )


def get(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
