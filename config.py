from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings for Aurelium AI Copilot."""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    app_name: str = "Aurelium AI Research Copilot"
    environment: str = "development"
    debug: bool = True

    # Database & Vector Store
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/aurelium",
        description="Postgres connection string with pgvector"
    )
    embedding_dimension: int = 768

    # ML Microservice
    use_ml_stubs: bool = Field(
        default=True,
        description="If True, uses in-memory stub implementations instead of calling ML microservice"
    )
    ml_service_url: str = "http://localhost:8001"

    # LLM Settings
    openai_api_key: str = Field(default="sk-placeholder", description="OpenAI or compatible LLM key")
    default_model: str = "gpt-4o"
    temperature: float = 0.2


settings = Settings()
