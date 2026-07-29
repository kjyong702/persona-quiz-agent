from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """환경변수 설정.

    Phase 2 범위에서 필요한 값은 DB 경로뿐이다. API 키와 판정 임계값은
    실제로 쓰는 Phase 3에서 추가한다. 쓰지 않는 설정을 미리 열어두면
    "설정에는 있는데 아무 데도 안 쓰이는 값"이 쌓인다.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "sqlite+aiosqlite:///./quiz.db"


settings = Settings()
