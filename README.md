# persona-quiz-agent

페르소나 AI 호스트가 진행하는 상식 퀴즈 서버입니다.

이 리포는 두 가지를 보여주기 위한 프로젝트입니다.

1. **문서 주도 AI 개발 프로세스** — 구현 전에 설계 문서(MVP 정의, DB 스키마, API 스펙)를 먼저 합의하고 짧은 문서를 AI 코딩 도구의 컨텍스트로 사용해 개발하는 방식입니다. 회사에서 설계해 팀 표준으로 정착시킨 프로세스를 개인 프로젝트에 처음부터 다시 설계해 적용했습니다. 커밋 로그를 보면 문서가 항상 코드보다 먼저입니다.
2. **LLM 하이브리드 판정과 평가 파이프라인** — 자연어 답변을 임베딩 유사도로 1차 판정하고 애매한 구간만 LLM에 넘기는 2단계 판정 구조, 그리고 promptfoo 기반으로 판정 정확도와 페르소나 응답 품질을 회귀 테스트하는 평가 체계입니다.

## 동작 개요

- 퀴즈 세트를 고르고 세션을 시작하면 페르소나 호스트(이름, 성격, 말투를 가진 AI)가 문제를 내고 답변을 판정하며 리액션합니다
- 정답 판정: 답변 임베딩과 기대 정답 임베딩의 코사인 유사도로 명확한 구간을 즉시 판정, 애매한 구간만 LLM 2차 판정, 임베딩 실패 시 LLM 단독 폴백
- 페르소나: 프로필 스키마에서 시스템 프롬프트를 자동 생성

## 문서

| 문서 | 내용 |
|---|---|
| [CLAUDE.md](CLAUDE.md) | AI 코딩 도구용 프로젝트 규칙 (설정 + 공통 규칙) |
| [docs/mvp.md](docs/mvp.md) | 문제 정의와 범위 |
| [docs/db-schema.md](docs/db-schema.md) | DB 스키마 |
| [docs/api-spec.md](docs/api-spec.md) | API 스펙 |
| [docs/dev-plan.md](docs/dev-plan.md) | 단계별 개발 계획 |
| [docs/design/python-backend-rules.md](docs/design/python-backend-rules.md) | 백엔드 설계 규칙 (레이어링, 네이밍, 에러 포맷) |
| [docs/_templates/](docs/_templates/) | 다음 프로젝트에 복사해 쓰는 마스터 템플릿 |

## 스택

Python 3.12, FastAPI, SQLite(SQLAlchemy), ChromaDB, OpenAI API, promptfoo
