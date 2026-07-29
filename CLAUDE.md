# persona-quiz-agent

## Part 1. 프로젝트 설정

- 서비스: 페르소나 AI 호스트가 진행하는 상식 퀴즈 서버
- 스택: Python 3.12, FastAPI, SQLAlchemy(SQLite), ChromaDB, OpenAI API
- 실행: `uvicorn app.main:app --reload` / 테스트: `pytest` / 평가: `npx promptfoo eval`
- 환경변수: `.env` (OPENAI_API_KEY, 판정 임계값 등) — `.env.example` 참고

### 개발 프로세스 (이 리포의 핵심 규칙)

1. **문서가 코드보다 먼저다.** 기능 구현 전에 docs/의 해당 문서(mvp, db-schema, api-spec)가 먼저 합의되어야 한다
2. 스펙이 바뀌면 **문서를 먼저 수정**하고 코드를 따라 바꾼다. 문서와 코드가 다르면 문서가 거짓말이 되고 이 체계 전체가 무너진다
3. 구현 순서는 `docs/dev-plan.md`의 Phase를 따른다. Phase를 건너뛰거나 합치지 않는다
4. 컨텍스트는 짧은 문서 단위로 준다. 전체 코드베이스를 읽히는 것은 최후 수단

## Part 2. 공통 규칙

### 레이어 구조

```
app/
  main.py          앱 초기화, 라우터 등록
  routers/         HTTP 입출력만 (검증은 스키마, 로직은 서비스에 위임)
  services/        비즈니스 로직 (판정 파이프라인, 페르소나 생성 포함)
  repositories/    DB 접근 (SQLAlchemy 쿼리와 커밋은 여기만)
  models/          SQLAlchemy 테이블 정의 (db-schema.md와 1대1)
  schemas/         Pydantic 요청/응답 모델
  core/            설정, 예외, 공용 클라이언트 (OpenAI, ChromaDB)
```

- 의존 방향은 routers -> services -> repositories -> models 단방향. 역방향 import 금지
- `schemas/`와 `core/`는 층이 아니라 **공용 어휘**다. 어느 층에서든 import해도 된다. 층 사이로 데이터를 넘길 때는 각자 정의한 타입 대신 schemas의 모델을 쓴다
- **ORM 객체는 서비스 밖으로 나가지 않는다.** 서비스가 모델을 스키마로 변환해 리턴하고 라우터는 스키마만 다룬다. 라우터가 ORM 객체를 받으면 지연 로딩이 응답 직렬화 시점에 터지고 내부 컬럼이 그대로 노출된다
- 외부 API(OpenAI) 호출은 core의 클라이언트 모듈을 거친다. 서비스에서 SDK 직접 호출 금지 (폴백/재시도를 한 곳에서 관리)

### 네이밍

- 파일/함수/변수: snake_case, 클래스: PascalCase
- 라우터 파일은 리소스 복수형 (quizzes.py, sessions.py), 서비스는 역할 명사 (judge_service.py, persona_service.py)
- DB 테이블: 복수형 snake_case, PK는 `id`, FK는 `<단수형>_id`

### API 규칙

- 응답은 공통 래퍼 `{ "data": ..., "error": null }` / 에러 시 `{ "data": null, "error": { "code", "message" } }`
- 에러 코드는 `core/exceptions.py`에 상수로 모은다. 문자열 직접 리턴 금지
- 상태 코드: 조회 200, 생성 201, 검증 실패 422(FastAPI 기본), 도메인 에러 400, 없음 404

### 테스트

- 서비스 레이어 단위 테스트 필수 (판정 파이프라인은 임계값 경계 케이스 포함)
- 외부 API는 목으로 대체. 실제 호출 테스트는 promptfoo 평가로 분리
