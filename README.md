# persona-quiz-agent

페르소나 AI 호스트가 진행하는 상식 퀴즈 서버입니다.

## 이 서버는 무엇인가

**자율 판단형 agent가 아니라, 정해진 순서를 수행하는 워크플로우형 agent의 실행 서버입니다.**

LLM이 다음에 무엇을 할지 스스로 정하지 않습니다. 순서가 코드에 있습니다.

```
답변 수신 -> 정규화 -> 임베딩 -> 임계값 분기 -> (애매하면) LLM 판정 -> 페르소나 멘트
```

**그렇게 만든 이유가 둘입니다.**

| | |
|---|---|
| **일관성** | 같은 답변에 같은 판정이 나와야 합니다. 매번 경로가 달라지면 점수를 신뢰할 수 없습니다 |
| **검증 가능성** | 경로가 고정되어야 각 단계를 따로 잴 수 있습니다. 임계값을 372건으로 정하고 프롬프트를 버전별로 비교한 것이 전부 이 구조 덕분입니다 |

도구를 고르고 계획을 세우는 자율형이 필요한 문제가 아니었습니다. **판정의 정확도와
비용이 문제였고, 그건 경로를 고정해야 잴 수 있습니다.**

## 무엇을 보여주는 리포인가

1. **문서 주도 AI 개발 프로세스** — 구현 전에 설계 문서(MVP 정의, DB 스키마, API 스펙)를
   먼저 합의하고 짧은 문서를 AI 코딩 도구의 컨텍스트로 씁니다. 회사에서 설계해 팀 표준으로
   정착시킨 프로세스를 개인 프로젝트에 처음부터 다시 설계해 적용했습니다.
   **커밋 로그를 보면 문서가 항상 코드보다 먼저입니다.**
2. **LLM 하이브리드 판정과 평가 파이프라인** — 임베딩으로 1차 판정하고 애매한 구간만
   LLM에 넘기는 구조, 그리고 그 임계값을 **감이 아니라 실측으로** 정한 과정입니다.

## 판정이 갈리는 기준

| 조건 | 경로 | LLM 호출 |
|---|---|---|
| 유사도 >= **0.92** | `embedding` (정답 확정) | 없음 |
| 유사도 <= **0.24** | `embedding` (오답 확정) | 없음 |
| 그 사이 | `llm` | 1회 |
| **상한을 넘겼지만 부정 표현이 있음** | `llm` | 1회 |
| 임베딩이나 벡터 조회 실패 | `fallback` | 1회 |
| 인덱스가 다른 모델로 만들어짐 | **판정하지 않고 503** | |
| 양쪽 다 실패 | 판정하지 않고 503 | |

**부정 표현을 따로 빼는 이유**: 임베딩은 뜻이 아니라 표현을 잽니다.
`"세종대왕 아닙니다"`가 유사도 0.926으로 정답들보다 높게 나옵니다.
규칙이 판정하지는 않고 **경로만 바꿉니다.** 반어법이나 전언은 정답일 수 있어서
최종 판단은 LLM이 합니다.

**판정 레코드에는 결과가 아니라 근거를 남깁니다.** 유사도, 다른 문제와의 유사도,
임베딩 모델 ID, 정규화 템플릿 버전이 함께 저장됩니다. **유사도 0.83은 그 자체로
의미가 없고** "어떤 모델로, 어떤 문자열 형태로 재었는가"가 있어야 값이 됩니다.

## 측정 결과

라벨링 평가셋 **372건**으로 잰 값입니다. 전 과정은 [eval/README.md](eval/README.md).

| 경로 | 건수 | 오류 |
|---|---|---|
| 임베딩 확정 | 106 (28.5%) | **0** |
| LLM 위임 | 266 (71.5%) | 7 |
| **전체** | **372** | **7 (98.1%)** |

> **"오류 0"은 임베딩이 확정한 28.5%에 대한 이야기입니다.** 시스템 전체로는 98.1%고,
> 둘을 구분하지 않으면 성능을 과장하게 됩니다.

**판정 프롬프트는 실측으로 세 번 고쳤습니다.**

| 버전 | 정확도 | 오답을 정답이라 함 | 정답을 오답이라 함 |
|---|---|---|---|
| judge.v1 | 94.4% | **4** | 11 |
| **judge.v3** (운영) | **97.4%** | **1** | 6 |

**`temperature=0`은 결정성을 보장하지 않습니다.** 같은 프롬프트로 세 번 돌리면
259, 260, 259가 나옵니다. **1~2건 차이는 개선이라고 부르지 않습니다.**
원인 추적과 대응은 [docs/notes/determinism.md](docs/notes/determinism.md).

## 실행

[uv](https://docs.astral.sh/uv/)로 의존성을 관리합니다.

```bash
uv sync                          # 의존성 설치 (Python 3.12 자동 설치)
cp .env.example .env             # OPENAI_API_KEY를 채웁니다
uv run python -m scripts.seed    # 퀴즈 세트, 페르소나, 앵커 임베딩 적재
uv run uvicorn app.main:app --reload
```

API 문서는 http://127.0.0.1:8000/docs 에 있습니다.

### 컨테이너로

```bash
export OPENAI_API_KEY=sk-...
docker compose up -d --build
docker compose exec api python -m scripts.seed   # 앵커 적재 (볼륨에 남습니다)
curl localhost:8000/readyz
```

**앵커 적재를 자동화하지 않았습니다.** 부팅 훅으로 옮기면 편하지만
**매 배포마다 임베딩 비용이 납니다.** 명시적으로 돌리는 쪽을 택했습니다.

### 헬스체크

| | 무엇을 묻나 | 실패하면 |
|---|---|---|
| `/healthz` | 프로세스가 **살아 있나** | 재시작 |
| `/readyz` | **일할 수 있나** (앵커, 인덱스 도장, 키) | 트래픽만 안 보냄 |

**재시작으로 고쳐지지 않는 문제는 liveness가 아닙니다.** 앵커가 없다고 재시작하면
같은 상태로 다시 떠서 루프에 빠집니다. Docker `HEALTHCHECK`도 `/healthz`를 씁니다.

### 테스트와 평가

```bash
uv run pytest                                              # 193건
uv run python -m scripts.analyze --sweep --holdout         # 임계값 (무료, 캐시 사용)
npx promptfoo@latest eval -c eval/promptfoo/judge.yaml     # 판정 프롬프트 (유료)
npx promptfoo@latest eval -c eval/promptfoo/persona.yaml --repeat 10   # 페르소나 회귀
```

**페르소나 평가에 `--repeat`이 필요한 이유**: 진행 멘트는 `temperature=0.8`입니다.
매번 같은 문장이면 진행자가 녹음기처럼 들려서 **일부러 흔들어뒀습니다.**
한 번 통과는 아무것도 뜻하지 않고 10회 중 10회여야 합니다.

## 스택

Python 3.12, FastAPI, SQLite(SQLAlchemy + aiosqlite), ChromaDB, OpenAI API,
structlog, promptfoo, Docker

**DB 접근을 비동기로 둔 것은 취향이 아니라 요구사항입니다.** 이 서버의 병목은 자체
연산이 아니라 외부 LLM 호출이고, 그 흐름을 동시 인플라이트 상한과 레이트 리미터로
제어하는 것이 이 프로젝트의 과제 중 하나입니다. 요청 경로 중간에 동기 DB 호출이
섞이면 이벤트 루프가 막혀 그 제어가 무의미해집니다.

## 알려진 한계

**MVP 범위라 의도한 것들입니다.**

- **앱 인스턴스를 늘릴 수 없습니다.** SQLite는 DB 서버가 아니라 라이브러리라
  앱 프로세스가 파일을 직접 만집니다. 볼륨을 공유하면 잠금 경합이 나고
  (실측: 두 번째 컨테이너가 테이블 생성조차 실패), 안 공유하면 각자 다른 DB를 봅니다.
  늘리려면 DB를 밖으로 빼야 합니다
- **계측이 프로세스 로컬입니다.** 워커를 늘리면 `/metrics`가 인스턴스마다 다릅니다
- **판정 로그에 토큰 사용량이 없습니다.** 비용이 설계 근거인데 실제 소모량을 안 남깁니다
- **마이그레이션 도구가 없습니다.** `create_all`은 기존 테이블을 안 고칩니다.
  운영 데이터가 생기는 시점에 Alembic으로 옮깁니다
- **임베딩은 뜻이 아니라 표현을 잽니다.** 같은 뜻의 답변이 쌍 안에서 유사도가
  평균 0.495 벌어집니다. 크로스 인코더나 NLI를 중간에 넣는 3단 구조가 다음 수이고
  [docs/dev-plan.md](docs/dev-plan.md) Phase 8에 계획으로 적어뒀습니다

## 문서

| 문서 | 내용 |
|---|---|
| [CLAUDE.md](CLAUDE.md) | AI 코딩 도구용 프로젝트 규칙과 커밋 규약 |
| [docs/mvp.md](docs/mvp.md) | 문제 정의와 범위 |
| [docs/db-schema.md](docs/db-schema.md) | DB 스키마 |
| [docs/api-spec.md](docs/api-spec.md) | API 스펙 |
| [docs/dev-plan.md](docs/dev-plan.md) | 단계별 개발 계획 |
| [docs/design/python-backend-rules.md](docs/design/python-backend-rules.md) | 백엔드 설계 규칙 |
| [eval/README.md](eval/README.md) | **평가 결과 전체** |

### 측정 기록

| 문서 | 무엇을 쟀나 |
|---|---|
| [threshold-measurement.md](docs/notes/threshold-measurement.md) | 임계값을 6차에 걸쳐 확정한 과정 |
| [determinism.md](docs/notes/determinism.md) | `temperature=0`인데 흔들리는 것과 그 대응 |
| [prompt-caching.md](docs/notes/prompt-caching.md) | 캐싱이 걸리는 조건과 문턱 역산 |
| [concurrency.md](docs/notes/concurrency.md) | 동시성과 레이트 리밋, 429 재현 |
| [index-drift.md](docs/notes/index-drift.md) | 모델이 바뀌면 조용히 틀리는 것 |
| [persona-regression.md](docs/notes/persona-regression.md) | 말투와 정답 유출 회귀 |
| [duplicate-submit.md](docs/notes/duplicate-submit.md) | 중복 제출과 DB 안전망 |
| [logging.md](docs/notes/logging.md) | 조용히 삼킨 것에 흔적 남기기 |
| [deployment.md](docs/notes/deployment.md) | 타임아웃 체인, 헬스체크, 컨테이너 |
| [judge-normalization.md](docs/notes/judge-normalization.md) | 판정 입력 정규화와 템플릿 버전 |
| **[build-log.md](docs/notes/build-log.md)** | **개발하면서 걸린 것들** (터진 것, 예방한 것, 발견한 구멍) |

`docs/_templates/`는 다음 프로젝트에 복사해 쓰는 마스터 템플릿입니다.

## 진행 상태

| Phase | 내용 | 상태 |
|---|---|---|
| 0, 1 | 프로세스 자산과 설계 문서 | 완료 |
| 2 | 스캐폴드, 퀴즈/세션 API | 완료 |
| 3 | 하이브리드 판정 파이프라인 | 완료 |
| 3.5 | 동시성과 레이트 리밋 | 완료 (실측 포함) |
| 4 | 페르소나 레이어와 프롬프트 캐싱 | 완료 |
| 5 | 평가 파이프라인 | 완료 |
| 6 | 배포 구성 | 완료 |
| 7 | 마무리 | 완료 |
| 8, 9 | 판정 3단 구조, 되묻기 | **계획만** ([dev-plan](docs/dev-plan.md)) |
