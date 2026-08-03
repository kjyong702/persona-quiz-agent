# 로깅 — 조용히 삼킨 것에 흔적을 남긴다

> 재현: `uv run pytest tests/test_log.py`
> 실제 출력을 보려면 `LOG_JSON=false uv run uvicorn app.main:app`

## 왜 필요했나

**예외를 폴백으로 흡수하면 흔적이 남지 않았다.** 임베딩이 죽어도 퀴즈는 LLM으로
계속 돈다. 그건 옳은 동작인데 **밖에서 보면 아무 일도 없다.**

사후에 "왜 그날 LLM 비용이 두 배였지"를 물으면 답할 근거가 없었다.
로깅이 **한 줄도 없었다.**

## 로깅과 트레이싱은 다르다

**로깅은 일어난 일을 그때그때 적고, 트레이싱은 요청 문맥을 단계들에 걸쳐 보존한다.**

답변 하나를 판정하는 데 임베딩 1회 + LLM 0~1회가 나간다. 로그가 흩어져 있으면
**어느 판정에 속한 호출인지 알 수 없다.**

여기서는 **요청 ID를 contextvar에 심어 모든 로그에 자동으로 붙이는** 정도만 했다.
OpenTelemetry 같은 분산 추적은 안 붙였다. 서비스가 하나뿐이라 경계를 넘는 전파가
필요 없고, 도구 도입 자체가 이 프로젝트의 주제가 아니다.

**앞단이 `X-Request-ID`를 보내면 그것을 이어받는다.** 프록시를 두면 보통 거기서
붙여 보내고, 그래야 프록시 로그와 우리 로그가 이어진다.

```
[http.request] path: /personas    request_id: from-proxy-abc     <- 앞단이 준 값
[http.request] path: /quiz-sets   request_id: 89ae8652aa48       <- 없으면 새로 만든다
```

## 무엇을 남기는가

### 부팅 한 줄 — 이 인스턴스가 무엇을 들고 도는가

```json
{"event": "app.started", "embedding_model": "text-embedding-3-small",
 "judge_model": "gpt-4o-mini-2024-07-18", "judge_prompt": "judge.v3",
 "upper_threshold": 0.92, "lower_threshold": 0.24, "api_key_fingerprint": "3f2db68a"}
```

**배포된 인스턴스가 어느 모델과 어느 프롬프트로 도는지는 로그 첫 줄에서 확인할 수
있어야 한다.** 컨테이너를 여러 개 띄우면 그중 하나만 옛 설정으로 도는 일이 생긴다.

`api_key_fingerprint`는 sha256 앞 8자다. **키 자체는 절대 안 남긴다.**
지문이면 회전이 반영됐는지 구분하기에 충분하고 원본은 복원되지 않는다.

### 판정 한 건 — 전체 이야기가 한 줄에

```
judge.completed
  question_id, method, is_correct, similarity, rival_similarity,
  embedding_model, template_version, judge_prompt, judge_model, duration_ms
```

**`judge_prompt`가 핵심이다.** 프롬프트는 코드와 다른 주기로 바뀌므로
**커밋 해시만으로는 어느 버전이 그 판정을 냈는지 알 수 없다.** `judge.v1`이 낸
판정과 `judge.v3`이 낸 판정이 로그에서 갈려야 프롬프트를 바꾼 효과를 사후에 본다.

이건 배포 단위가 갈라진다는 이야기의 실물이기도 하다. 코드와 프롬프트가 따로
움직이면 **로그에도 따로 찍혀야 한다.**

### 삼키던 자리마다

| 자리 | 이벤트 | 왜 |
|---|---|---|
| 임베딩/벡터 장애 | `judge.fallback` (`embedding_unavailable`) | 폴백은 옳지만 조용하면 안 된다 |
| 앵커 없음 | `judge.fallback` (`no_anchor`) | **설정 실수일 가능성이 높다** |
| 정규화가 비운 답변 | `judge.fallback` (`empty_after_normalization`) | 제공사는 멀쩡한데 우리가 못 보낼 입력을 보낸 것. 외부 장애로 세면 지표가 오염된다 |
| 인덱스 드리프트 | `judge.index_drift` (error) | 구성 오류 |
| `logprobs` 파싱 실패 | `judge.logprobs_unavailable` | **불안정 관측이 통째로 죽어도 지표는 0으로 보인다** |
| `Retry-After` 파싱 실패 | `retry.header_unparsable` | 헤더 형식이 바뀐 것을 영영 모르게 된다 |

마지막 둘이 특히 그렇다. **관측 장치가 고장 나면 "문제 없음"과 구분이 안 된다.**

## 무엇을 안 남기는가

**답변 원문과 프롬프트 전문을 남기지 않는다.** 사용자 입력이고 로그는 보통
평문으로 여러 곳에 복제된다. 어느 경로로 갔고 유사도가 얼마였는지만 있으면
비용과 품질 추적에는 충분하다.

테스트로 확인한다. 주민번호가 든 답변을 판정시키고 캡처한 로그에 그 문자열이
없는지 본다. **"안 남긴다"는 주석은 지켜지는지 확인할 방법이 없으면 의미가 없다.**

## structlog을 고른 이유

표준 라이브러리 `logging`으로도 JSON은 만들 수 있다. `logging.Formatter`를 상속해
`format()`에서 `json.dumps`를 부르면 된다. 그래도 `structlog`을 쓴 이유가 셋이다.

**첫째, 값을 키워드로 넘긴다.**

```python
logging.info("판정 완료 method=%s similarity=%s", method, sim)   # 문자열을 조립
_log.info("judge.completed", method=method, similarity=sim)      # 값을 그대로
```

앞은 사람이 읽을 문장을 만들고 나중에 파싱해야 한다. 뒤는 처음부터 구조다.
**필드를 추가해도 문자열 포맷을 안 건드린다.**

**둘째, `contextvars`를 프로세서로 지원한다.** 요청 ID를 한 번 심으면 이후 모든
로그에 자동으로 붙는다. 표준 라이브러리로 하려면 `LogRecord`에 필드를 넣는
필터를 직접 만들어야 한다.

**셋째, 프로세서 체인이 리스트다.** 타임스탬프, 로그 레벨, 예외 트레이스백,
JSON 렌더링이 순서 있는 함수 목록이라 **무엇이 어떤 순서로 붙는지가 코드에 보인다.**
`format_exc_info`를 넣고 빼는 것으로 트레이스백 포함 여부가 갈린다.

모듈 이름은 `app/core/log.py`다. `logging.py`로 지으면 표준 라이브러리와 헷갈린다.
같은 파일에서 둘 다 쓰기 때문에 이름이 겹치면 읽는 사람이 매번 멈춘다.

## 남은 것

- **계측(`metrics.py`)은 여전히 프로세스 로컬이다.** 워커를 늘리면 `/metrics`가
  인스턴스마다 다른 값을 준다. 로그는 수집하면 합쳐지지만 계측은 아니다
- **토큰 사용량을 판정 로그에 안 넣었다.** `generate_host_message`는 계측에
  남기는데 `judge_answer`는 아니다. 비용이 설계 근거인데 실제 소모량을 안 남긴다
- 로그 표본 추출이 없다. 트래픽이 늘면 판정마다 한 줄이 부담이 된다
- 분산 추적(OpenTelemetry)은 안 붙였다
