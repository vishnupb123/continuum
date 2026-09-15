from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://continuum:continuum@postgres:5432/continuum"
    redis_url: str = "redis://redis:6379/0"
    cors_origins: list[str] = ["http://localhost:3000"]
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
