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
    upper_threshold: float = 0.82
    lower_threshold: float = 0.55
    # 해당 문제 앵커와 다른 문제 앵커의 유사도 차이 하한.
    # 이 값보다 좁으면 상한을 넘겨도 즉시 정답 처리하지 않고 LLM으로 넘긴다
    min_margin: float = 0.05


settings = Settings()
