from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "sqlite+aiosqlite:///./quiz.db"

    # 커넥션 풀. 기본값(5+10)을 그대로 두면 **이 값이 동시 LLM 호출 상한이 된다.**
    # 요청 스코프 세션이 외부 호출을 기다리는 내내 커넥션을 쥐고 있기 때문이다.
    # 부하 실험에서 실측했다. 풀 15 / LLM 지연 1.35초 -> 처리량 11.1건/초로
    # 정확히 묶였고, 동시성을 25에서 120으로 올려도 지연만 3배가 됐다.
    # 자세한 것은 docs/notes/concurrency.md
    db_pool_size: int = 20
    db_max_overflow: int = 40

    # --- 외부 API ---
    openai_api_key: str | None = None
    # 모델은 별칭이 아니라 날짜가 박힌 스냅샷으로 고정한다.
    # 별칭은 제공사가 가리키는 대상을 바꿀 수 있고, 그러면 예고 없이
    # 판정 기준이 달라진다. 모델을 바꿀 때는 평가를 다시 돌린다
    embedding_model: str = "text-embedding-3-small"
    judge_model: str = "gpt-4o-mini-2024-07-18"
    llm_timeout_seconds: float = 20.0

    # 판정에만 시드를 건다. 멘트는 매번 달라야 하므로 걸지 않는다.
    # temperature와 같은 이유로 갈리는 설정이다.
    #
    # **시드는 결정성을 보장하지 않는다.** 제공사도 "mostly deterministic"이라고만
    # 말한다. temperature 0으로도 266건 중 1건이 흔들렸고(재현: --repeat),
    # 시드를 넣어 그 값이 줄어드는지 실측한 결과는 docs/notes/determinism.md에 있다.
    #
    # 근본 원인은 난수가 아니라 부동소수점이다. 같은 입력이라도 서버가 다른
    # 요청들과 함께 배치로 묶으면 행렬 연산의 누적 순서가 달라지고, 최상위 두 토큰의
    # 확률이 붙어 있을 때 순위가 뒤집힌다. 클라이언트가 통제할 수 있는 변수가 아니다
    judge_seed: int | None = 20260803

    # 판정이 흔들릴 자리를 **미리 알아내는** 구간이다. logprobs로 correct 토큰의
    # 확률을 받아 이 사이에 들어오면 불안정으로 기록한다.
    #
    # 경계 19건을 20회씩 돌려 정한 값이다. 결과가 갈린 케이스는 관측 확률이
    # 0.047~0.818을 오갔고, 갈리지 않은 케이스는 0.0001 이하이거나 0.92 이상에
    # 붙어 있었다. 측정은 docs/notes/determinism.md.
    #
    # **임베딩 층에서 걷어낸 margin과 같은 발상인데 여기서는 실제로 발화한다.**
    # 임베딩에서는 다른 문항 앵커가 붙는 일이 없어 372건 중 0번이었지만,
    # LLM은 애매한 답변에서 두 판정의 확률이 실제로 붙는다
    unstable_low: float = 0.02
    unstable_high: float = 0.92

    # 진행 멘트는 판정과 설정이 반대다. 판정은 재현성이 전부라 temperature 0이고,
    # 멘트는 매번 같은 문장이면 녹음기처럼 들려서 흔들림을 남긴다
    host_temperature: float = 0.8
    host_max_tokens: int = 200

    # 자격증명을 다시 읽는 주기. 근거는 app/core/credentials.py 문서 문자열.
    # 0으로 두면 매 호출마다 환경을 읽는다. 값이 같으면 클라이언트는 재사용하므로
    # 비용은 환경변수 조회 한 번이지만, 기본은 5분으로 둔다
    credential_ttl_seconds: float = 300.0

    # 로깅. 프로덕션은 JSON이어야 수집기가 파싱한다.
    # 개발에서 읽기 힘들면 LOG_JSON=false로 끈다
    log_json: bool = True
    log_level: str = "INFO"

    chroma_path: str = "./.chroma"

    # --- 나가는 호출 흐름 제어 ---
    # 두 스위치를 따로 둔 이유는 부하 실험 때문이다. 코드를 고치지 않고
    # 설정만 뒤집어야 대응 전후를 같은 코드로 잰 값이 된다.
    # 둘을 조합하면 흐름 정형과 실패 흡수가 각각 얼마나 기여했는지도 갈린다
    gate_enabled: bool = True
    retry_enabled: bool = True

    # 쿼터는 모델과 엔드포인트마다 따로 걸리므로 게이트도 따로 둔다.
    # 이 값은 실제 쿼터보다 **낮게** 잡아야 의미가 있다.
    # 한도와 같게 두면 우리 쪽 계산 오차만큼 그대로 429가 난다
    embedding_max_concurrency: int = 8
    embedding_rpm: float = 2000
    llm_max_concurrency: int = 4
    llm_rpm: float = 300

    retry_max_attempts: int = 4
    retry_initial_delay: float = 0.5
    retry_max_delay: float = 20.0

    # --- 판정 임계값 ---
    # 라벨링 평가셋 372건의 실측 분포로 정한 값이다. 근거와 과정은
    # docs/notes/threshold-measurement.md, 재현은 scripts/analyze.py.
    #
    # 기준은 **false accept(오답을 정답으로 확정) 0건**이다. 이 오류는 LLM이
    # 고칠 기회조차 없어서 LLM 호출 비용으로 살 수 있는 종류가 아니다.
    # 그 조건에서 false reject까지 0으로 만드는 가장 싼 조합을 골랐고,
    # 대가는 LLM 위임 71.1%다. 비용보다 정확도를 먼저 둔 선택이다.
    #
    # 문항 단위 홀드아웃(`--holdout`)에서 상한은 train과 전체가 0.90으로 일치했다.
    # 하한은 데이터에 민감해(0.26 vs 0.38) 안전한 쪽인 낮은 값을 택했다.
    #
    # **두 값 모두 최적값이 아니라 여유를 얹은 값이다.** 최적값은 이 데이터에서만
    # 오류 0이고, 경계에 가장 가까운 샘플이 하나만 더 나와도 깨진다.
    #
    #   상한  최적 0.90 -> 0.92   부정문 제외 오답 최대가 0.893이라 여유 0.007뿐
    #   하한  최적 0.26 -> 0.24   정답 최소가 0.264라       여유 0.004뿐
    #
    # 여유를 각각 0.027과 0.024로 벌리는 대가는 LLM 위임 +2.7%p와 +1.6%p다.
    # "오류 0"이라는 판단을 새 데이터에서도 유지하려면 치러야 하는 값이다.
    #
    # 하한을 아예 없애면 false reject가 구조적으로 0이 되지만 그렇게는 안 했다.
    # 하한이 아끼는 호출이 7.3%로 작긴 해도, 없애면 판정이 "확정 아니면 위임"의
    # 2분기가 되어 **오답을 오답으로 확정하는 경로 자체가 사라진다.**
    #
    # **처음 보는 문항에서는 FR이 소수 나온다.** "이 데이터에서 0"이지 "항상 0"이 아니다
    upper_threshold: float = 0.92
    lower_threshold: float = 0.24
    # min_margin은 걷어냈다. 평가셋 372건에서 한 번도 발화하지 않았고, 근거로 삼던
    # 전제("몰라요" 류가 상한을 통과한다)도 반증됐다. 상세는
    # docs/notes/threshold-measurement.md. rival_similarity는 계속 기록하므로
    # 문제가 늘어 주제가 겹치면 그 수치를 보고 되살릴 수 있다


settings = Settings()
