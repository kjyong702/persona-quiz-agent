# DB 스키마

SQLite + SQLAlchemy. 임베딩 벡터는 ChromaDB에 별도 저장 (컬렉션 구성은 하단).

## personas

| 컬럼 | 타입 | 설명 |
|---|---|---|
| id | INTEGER PK | |
| name | TEXT | 호스트 이름 |
| personality | TEXT | 성격 한 줄 (예: 승부욕 강한 열혈) |
| speech_style | TEXT | 말투 규칙 (예: 반말, 감탄사 자주) |
| reaction_style | TEXT | 정답/오답 리액션 성향 |
| created_at | DATETIME | |

## quiz_sets

| 컬럼 | 타입 | 설명 |
|---|---|---|
| id | INTEGER PK | |
| title | TEXT | |
| description | TEXT | |
| created_at | DATETIME | |

## questions

| 컬럼 | 타입 | 설명 |
|---|---|---|
| id | INTEGER PK | |
| quiz_set_id | INTEGER FK | |
| order_no | INTEGER | 세트 내 출제 순서 |
| question_text | TEXT | |
| expected_answers | TEXT(JSON) | 기대 정답 표현 목록 (대표 정답 + 허용 변형) |

## quiz_sessions

| 컬럼 | 타입 | 설명 |
|---|---|---|
| id | INTEGER PK | |
| quiz_set_id | INTEGER FK | |
| persona_id | INTEGER FK | |
| current_order | INTEGER | 진행 중인 문제 순서 (0 = 시작 전) |
| status | TEXT | in_progress / finished |
| created_at | DATETIME | |

## session_answers

| 컬럼 | 타입 | 설명 |
|---|---|---|
| id | INTEGER PK | |
| session_id | INTEGER FK | |
| question_id | INTEGER FK | |
| answer_text | TEXT | 사용자 답변 원문 |
| is_correct | BOOLEAN | |
| judge_method | TEXT | embedding / llm / fallback |
| similarity | REAL NULL | **해당 문제의** 기대 정답과의 최고 유사도 (fallback이면 NULL) |
| rival_similarity | REAL NULL | **다른 문제의** 기대 정답과의 최고 유사도 (fallback이면 NULL) |
| embedding_model | TEXT NULL | 판정에 쓴 임베딩 모델 ID |
| template_version | TEXT NULL | 임베딩 입력 정규화 템플릿 버전 |
| created_at | DATETIME | |

> judge_method와 similarity를 남기는 이유: 판정 경로 분포와 임계값 튜닝 근거가 이 데이터에서 나온다. 평가 파이프라인의 원천 데이터.

**rival_similarity를 함께 남기는 이유**: 절대 임계값 하나로만 판정하면 짧고 흔한 답변이 모든 문제에서 높은 유사도를 받아 통과한다. "이 문제의 정답과 얼마나 가까운가"만이 아니라 "다른 문제의 정답보다 얼마나 더 가까운가"가 필요하고, 그 차이(margin)를 튜닝하려면 두 값이 다 남아야 한다.

**embedding_model과 template_version을 남기는 이유**: 유사도 0.83은 그 자체로 아무 의미가 없고 "어떤 모델로, 어떤 문자열 형태로 재었는가"가 있어야 값이 된다. 모델이나 템플릿을 바꾸면 그 이전 유사도와는 비교할 수 없다. 이 두 컬럼이 없으면 임계값을 다시 정할 때 과거 데이터를 쓸 수 있는지 판단할 근거가 사라진다.

## ChromaDB 컬렉션

`expected_answers`: 문제별 기대 정답 표현의 임베딩.

| 항목 | 값 |
|---|---|
| 거리 함수 | 코사인 (`hnsw:space = cosine`). 유사도 = 1 - 거리 |
| 문서 ID | `q{question_id}-a{index}` |
| 문서 본문 | 정규화 템플릿을 적용한 기대 정답 문자열 |
| 메타데이터 | `question_id`, `quiz_set_id`, `raw_text`(정규화 전 원문), `embedding_model`, `template_version` |

- 임베딩은 우리가 직접 만들어 넣는다. Chroma의 기본 임베딩 함수를 쓰지 않는 이유는 판정에 쓰는 모델을 코드에서 명시적으로 고정하기 위해서다
- 시드 로드 시점에 일괄 임베딩
- 판정 시 조회는 두 번이다. `question_id` 필터로 해당 문제의 앵커, 필터 없이 전체에서 다른 문제의 앵커. 앞의 값이 similarity, 뒤의 값이 rival_similarity가 된다

> 데이터가 60벡터 규모라 벡터 DB가 성능상 필요한 크기는 아니다. 그럼에도 두는 이유는 메타데이터 필터와 거리 함수 설정을 코드가 아니라 스토어 구성으로 다루는 형태를 유지하기 위해서다. 규모가 커져도 판정 코드는 그대로 간다.
