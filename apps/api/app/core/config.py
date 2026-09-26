from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://continuum:continuum@postgres:5432/continuum"
    redis_url: str = "redis://redis:6379/0"
    cors_origins: list[str] = ["http://localhost:3000"]
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    JWT_SECRET_KEY: str = "dev-only-change-me"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    audio_storage_backend: str = "local"
    audio_storage_path: str = "/data/audio"
    transcription_provider: str = "mock"
    openai_api_key: str | None = None
    openai_transcription_model: str = "gpt-transcribe"
    local_whisper_model: str = "small.en"
    local_whisper_device: str = "cpu"
    local_whisper_compute_type: str = "int8"
    local_whisper_download_root: str = "/models"
    text_encoder_provider: str = "mpnet"

    text_encoder_model: str = (
      "sentence-transformers/all-mpnet-base-v2"
    )

    text_encoder_revision: str | None = None

    text_encoder_device: str = "cpu"

    text_encoder_cache_path: str = "/text-models"

settings = Settings()
