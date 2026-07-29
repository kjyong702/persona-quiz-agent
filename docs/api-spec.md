# API 스펙

공통 응답 래퍼와 에러 포맷은 CLAUDE.md Part 2를 따른다.

## 조회

### GET /personas
페르소나 목록. `[{ id, name, personality, speech_style }]`

### GET /quiz-sets
퀴즈 세트 목록. `[{ id, title, description, question_count }]`

## 세션

### POST /sessions
세션 시작. 호스트의 오프닝 멘트를 함께 반환한다.

- 요청: `{ "quiz_set_id": 1, "persona_id": 1 }`
- 응답 201: `{ "session_id", "host_message" }` — host_message는 페르소나가 생성한 인사말

### GET /sessions/{id}
진행 상태. `{ "status", "current_order", "total_questions", "correct_count" }`

- `current_order`는 **출제된 문제 번호**다. 0이면 아직 첫 문제를 받지 않은 상태

### POST /sessions/{id}/next
다음 문제 출제. 페르소나 말투로 문제를 소개한다.

- 응답 200: `{ "finished": false, "question_id", "order_no", "question_text", "host_message" }`
- 남은 문제가 없으면 세션을 finished로 바꾸고 `{ "finished": true, "host_message" }` (마무리 멘트 + 결과 요약). 이때 문제 필드들은 null
- **이미 출제한 문제가 아직 미답변이면 같은 문제를 다시 반환한다.** next를 연달아 호출해 문제를 건너뛰는 것을 막기 위한 규칙이고, 이 경우 `current_order`는 올라가지 않는다. 클라이언트 재시도에 안전하도록 멱등하게 둔다

### POST /sessions/{id}/answer
답변 제출과 판정. 이 프로젝트의 핵심 엔드포인트.

- 요청: `{ "answer": "자연어 답변" }`
- 응답 200:

```json
{
  "is_correct": true,
  "judge_method": "embedding",
  "similarity": 0.91,
  "host_message": "정답! (페르소나 리액션)"
}
```

- 판정 흐름:
  1. 답변을 **정규화 템플릿**에 통과시킨 뒤 임베딩. 기대 정답도 시드 시점에 같은 템플릿으로 임베딩되어 있다
  2. ChromaDB 조회 2회. `similarity` = 해당 문제 기대 정답과의 최고 코사인 유사도, `rival_similarity` = 다른 문제 기대 정답과의 최고 유사도
  3. `similarity >= UPPER_THRESHOLD` **그리고** `similarity - rival_similarity >= MIN_MARGIN`: 정답 확정 (LLM 호출 없음)
  4. `similarity <= LOWER_THRESHOLD`: 오답 확정 (LLM 호출 없음)
  5. 그 외(중간 구간, 또는 상한을 넘었지만 margin이 좁은 경우): LLM 2차 판정 (문제, 기대 정답, 사용자 답변을 주고 정오만 판단)
  6. 임베딩 또는 벡터 조회 실패: LLM 단독 판정 폴백 (`judge_method = fallback`, similarity는 null)
  7. LLM까지 실패: 판정하지 않고 503으로 실패시킨다. 틀린 판정을 조용히 내리는 것보다 낫다
- 임계값과 margin은 환경변수 (기본값은 평가 파이프라인 결과로 정한다 — dev-plan Phase 5)

**margin 조건을 둔 이유**: 절대 임계값만 두면 "몰라요", "네" 같은 짧고 흔한 답변이 어느 문제에서든 애매하게 높은 유사도를 받는다. 그런 답변은 이 문제의 정답과 가까운 만큼 다른 문제의 정답과도 가까우므로, 두 값의 차이를 보면 걸러진다. margin이 좁으면 즉시 판정하지 않고 LLM으로 넘긴다.

**응답에 없는 것**: 임베딩 모델 ID와 템플릿 버전은 DB에만 남기고 응답에 싣지 않는다. 판정 재현에 필요한 값이지 클라이언트가 알 것은 아니다.

## 에러

| 코드 | 상태 | 상황 |
|---|---|---|
| `SESSION_NOT_FOUND` | 404 | 없는 세션 |
| `QUIZ_SET_NOT_FOUND` | 404 | POST /sessions에 없는 quiz_set_id |
| `PERSONA_NOT_FOUND` | 404 | POST /sessions에 없는 persona_id |
| `QUIZ_SET_EMPTY` | 400 | 문제가 하나도 없는 세트로 세션 시작 |
| `SESSION_FINISHED` | 400 | 종료된 세션에 next 또는 answer |
| `NO_ACTIVE_QUESTION` | 400 | next 없이 answer, 또는 이미 답변한 문제에 다시 answer |
| `JUDGE_UNAVAILABLE` | 503 | 임베딩과 LLM이 모두 실패해 판정할 수 없음 |
