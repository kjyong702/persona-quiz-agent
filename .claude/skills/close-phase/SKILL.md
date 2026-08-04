---
name: close-phase
description: Phase를 닫기 전에 기록이 빠진 곳이 없는지 확인한다. 다음 단계로 넘어가려 할 때, "Phase가 끝났다"고 말하기 전에 사용
when_to_use: Phase 완료 선언 직전, 다음 Phase 착수 직전, 사용자가 "다음 단계로 가자"고 할 때
disable-model-invocation: true
argument-hint: [phase-number]
allowed-tools: Bash(grep:*) Bash(git log:*) Bash(ls:*) Bash(uv run pytest:*) Read
---

Phase $ARGUMENTS를 닫기 전 점검이다. **세지 않고 눈으로 훑지 않는다.**

## 지금 상태

- 마지막 커밋 5개: !`git log --oneline -5`
- 빌드 로그 항목 수: !`grep -c '^## [🔴🟡🔵⚪]' docs/notes/build-log.md`
- 측정 문서: !`ls docs/notes/*.md | wc -l`
- 테스트: !`uv run pytest -q --collect-only 2>/dev/null | grep -oE '[0-9]+ tests collected'`

## 기록처 셋 — 다 확인한다

**하나를 채우면 "기록했다"는 감각이 들어 나머지를 안 보게 된다.** 의지 문제가 아니라 구조 문제다.
셋은 성격이 달라서 서로를 대신하지 않는다.

1. **측정 결과와 판단 근거**가 `docs/notes/<주제>.md`에 있나
2. **이번 Phase에서 걸린 것**이 `build-log.md`에 성격 표시와 함께 있나
   - 🔴 터짐 / 🟡 예방 / 🔵 구멍 / ⚪ 감사
   - **성격을 섞으면 겪지 않은 일을 겪었다고 말하게 된다**
3. **README와 `.env.example`이 코드와 맞나**
   - 코드는 안 고치면 테스트가 깨지는데 **문서는 아무 일도 안 일어난다**
   - `.env.example`은 `Settings` 필드와 기계적으로 대조한다


## 다음 Phase 시작 전

4. **표준 용어나 알려진 패턴이 있는지 먼저 찾는다.** 만들고 나서 찾으면 늦다
   - 관례와 같으면 같다고, 다르면 왜 다른지를 `docs/notes/`에 적어둔다
5. **`build-log.md`의 🔵와 ⚪를 훑는다.** 이미 적어둔 구멍을 다시 발견하는 데 시간을 쓰지 않는다

## 보고

빠진 것이 있으면 **채우기 전에 먼저 목록으로 보고한다.** "복구했다"로 끝내지 않는다.
