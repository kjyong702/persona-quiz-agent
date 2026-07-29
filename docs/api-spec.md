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
  1. 답변 임베딩 -> ChromaDB에서 해당 문제의 기대 정답들과 코사인 유사도 계산
  2. 최고 유사도 >= UPPER_THRESHOLD: 정답 확정 (LLM 호출 없음)
  3. 최고 유사도 <= LOWER_THRESHOLD: 오답 확정 (LLM 호출 없음)
  4. 중간 구간: LLM 2차 판정 (문제, 기대 정답, 사용자 답변을 주고 정오만 판단)
  5. 임베딩 API 실패: LLM 단독 판정 폴백
- 임계값은 환경변수 (기본값은 평가 파이프라인 결과로 정한다 — dev-plan Phase 5)

## 에러

| 코드 | 상태 | 상황 |
|---|---|---|
| `SESSION_NOT_FOUND` | 404 | 없는 세션 |
| `QUIZ_SET_NOT_FOUND` | 404 | POST /sessions에 없는 quiz_set_id |
| `PERSONA_NOT_FOUND` | 404 | POST /sessions에 없는 persona_id |
| `QUIZ_SET_EMPTY` | 400 | 문제가 하나도 없는 세트로 세션 시작 |
| `SESSION_FINISHED` | 400 | 종료된 세션에 next 또는 answer |
| `NO_ACTIVE_QUESTION` | 400 | next 없이 answer, 또는 이미 답변한 문제에 다시 answer |
