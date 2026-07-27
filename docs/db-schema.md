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
| similarity | REAL NULL | 임베딩 최고 유사도 (fallback이면 NULL) |
| created_at | DATETIME | |

> judge_method와 similarity를 남기는 이유: 판정 경로 분포와 임계값 튜닝 근거가 이 데이터에서 나온다. 평가 파이프라인의 원천 데이터.

## ChromaDB 컬렉션

- `expected_answers`: 문제별 기대 정답 표현의 임베딩. 메타데이터로 question_id, 원문 저장
- 시드 로드 시점에 일괄 임베딩. 답변 판정 시 question_id 필터로 조회
