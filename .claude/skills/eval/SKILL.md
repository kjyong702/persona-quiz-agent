---
name: eval
description: 판정 프롬프트와 페르소나 회귀를 promptfoo로 평가한다. 유료 호출이라 순서를 틀리면 돈이 나간다
when_to_use: 프롬프트를 고친 뒤, 임계값을 바꾼 뒤, 페르소나 설정을 바꾼 뒤
disable-model-invocation: true
argument-hint: [judge|persona|both]
allowed-tools: Bash(uv run:*) Bash(npx promptfoo*) Bash(export:*) Read
---

평가 대상: $ARGUMENTS (없으면 물어본다)

## 먼저 확인 — 안 하면 다른 것을 재게 된다

```bash
export PROMPTFOO_PYTHON="$(pwd)/.venv/bin/python"   # 안 하면 시스템 파이썬을 써서 app을 못 찾는다
set -a && . ./.env && set +a                        # OPENAI_API_KEY
```

**임계값을 바꿨으면 케이스를 다시 만든다.** 위임 대상이 달라지기 때문이다.

```bash
uv run python -m scripts.build_promptfoo_cases
```

## 판정 프롬프트

```bash
npx promptfoo@latest eval -c eval/promptfoo/judge.yaml --no-cache -o eval/promptfoo/results-judge.json
uv run python -m scripts.summarize_promptfoo eval/promptfoo/results-judge.json
```

- 원본은 7MB가 넘어 **커밋하지 않는다.** 요약만 커밋한다
- **`--repeat` 없이 나온 1~2건 차이는 개선이 아니다.** `temperature=0`이어도 흔들린다

## 페르소나 회귀

```bash
npx promptfoo@latest eval -c eval/promptfoo/persona.yaml --no-cache --repeat 10
```

**`--repeat`이 필수다.** 진행 멘트는 `temperature=0.8`이라 일부러 흔들어뒀다.
**한 번 통과는 아무것도 뜻하지 않고 10회 중 10회여야 한다.**

설정을 바꿨으면 케이스를 다시 만든다.

```bash
uv run python -m scripts.build_persona_cases
```

## 통과했을 때 의심할 것

**전부 통과하면 검사가 작동하는지 확인해야 한다.** 통과만 하는 테스트는
검사하는 게 없는 것과 구분이 안 된다.

- 프롬프트를 일부러 깨뜨려 잡히는지 본다
- **앞쪽 금지 지시가 뒤에 붙인 지시를 이기므로**, 지시를 추가하는 방식으로는
  음성 대조가 안 될 수 있다. 프롬프트 전체를 갈아끼운다
- 확인 후 **반드시 원상 복구**한다

## 결과 기록

`eval/README.md`에 수치를 남긴다. 프롬프트를 올렸으면 `app/core/prompts.py`의
`JUDGE_PROMPT`도 같이 본다. **버전을 올려놓고 운영이 옛 버전을 쓰는 상태가 되기 쉽다.**
