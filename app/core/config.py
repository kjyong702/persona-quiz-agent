from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "sqlite+aiosqlite:///./quiz.db"

    # --- 외부 API ---
    openai_api_key: str | None = None
    # 모델은 별칭이 아니라 날짜가 박힌 스냅샷으로 고정한다.
    # 별칭은 제공사가 가리키는 대상을 바꿀 수 있고, 그러면 예고 없이
    # 판정 기준이 달라진다. 모델을 바꿀 때는 평가를 다시 돌린다
    embedding_model: str = "text-embedding-3-small"
    judge_model: str = "gpt-4o-mini-2024-07-18"
    llm_timeout_seconds: float = 20.0

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
    # Phase 5 평가 전의 임시값이다. 여기 적힌 숫자에는 아직 근거가 없고,
    # 라벨링 데이터셋으로 정확도와 LLM 호출 비율을 재서 확정한다
    # 라벨링 평가셋 372건의 실측 분포로 정한 값이다. 근거와 과정은
    # docs/notes/threshold-measurement.md, 재현은 scripts/analyze.py.
    #
    # 기준은 **false accept(오답을 정답으로 확정) 0건**이다. 이 오류는 LLM이
    # 고칠 기회조차 없어서 LLM 호출 비용으로 살 수 있는 종류가 아니다.
    # 그 조건에서 false reject까지 0으로 만드는 가장 싼 조합을 골랐고,
    # 대가는 LLM 위임 69.5%다. 비용보다 정확도를 먼저 둔 선택이다.
    #
    # 문항 단위 홀드아웃(`--holdout`)에서 상한은 train과 전체가 0.90으로 일치했다.
    # 하한은 데이터에 민감해(0.26 vs 0.38) 안전한 쪽인 낮은 값을 택했다.
    #
    # **상한은 최적값 0.90이 아니라 0.92로 올려 잡았다.** 부정문을 뺀 오답의 최대
    # 유사도가 0.893이라 0.90은 여유가 0.007뿐이다. 이 데이터에서만 오류 0이고
    # 새 문항에서 0.893보다 높은 오답이 하나만 나와도 깨진다. 여유를 0.027로
    # 벌리는 대가는 LLM 위임 +2.7%p이고, "오류 0"이라는 판단을 새 데이터에서도
    # 유지하려면 치러야 하는 값이다.
    #
    # **처음 보는 문항에서는 FR이 소수 나온다.** "이 데이터에서 0"이지 "항상 0"이 아니다
    upper_threshold: float = 0.92
    lower_threshold: float = 0.26
    # min_margin은 걷어냈다. 평가셋 372건에서 한 번도 발화하지 않았고, 근거로 삼던
    # 전제("몰라요" 류가 상한을 통과한다)도 반증됐다. 상세는
    # docs/notes/threshold-measurement.md. rival_similarity는 계속 기록하므로
    # 문제가 늘어 주제가 겹치면 그 수치를 보고 되살릴 수 있다


settings = Settings()
