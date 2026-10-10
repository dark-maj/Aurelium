from __future__ import annotations

import os
from typing import Optional
from pydantic import BaseModel, Field

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    _HAS_PYDANTIC_SETTINGS = True
except ImportError:
    BaseSettings = BaseModel  # type: ignore
    SettingsConfigDict = None  # type: ignore
    _HAS_PYDANTIC_SETTINGS = False


class Settings(BaseSettings):
    """
    Centralized settings for the Aurelium AI package.
    Reads from environment variables and root .env file.
    """
    if _HAS_PYDANTIC_SETTINGS:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
            populate_by_name=True,
        )

    # --- xAI / Grok Configuration ---
    xai_api_key: str = Field(default_factory=lambda: os.getenv("XAI_API_KEY", "xai-placeholder-key"), alias="XAI_API_KEY")
    xai_base_url: str = Field(default_factory=lambda: os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"), alias="XAI_BASE_URL")
    # Current flagship Grok model with native structured output & tool calling support
    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "grok-4.7"), alias="LLM_MODEL")
    judge_model: str = Field(default_factory=lambda: os.getenv("JUDGE_MODEL", "grok-4.7"), alias="JUDGE_MODEL")
    llm_timeout: float = Field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT", "60.0")), alias="LLM_TIMEOUT")
    llm_max_retries: int = Field(default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "3")), alias="LLM_MAX_RETRIES")

    # --- Pinecone Vector Database ---
    pinecone_api_key: str = Field(default_factory=lambda: os.getenv("PINECONE_API_KEY", "pc-placeholder-key"), alias="PINECONE_API_KEY")
    pinecone_index: str = Field(default_factory=lambda: os.getenv("PINECONE_INDEX", "aurelium-papers"), alias="PINECONE_INDEX")
    pinecone_namespace: str = Field(default_factory=lambda: os.getenv("PINECONE_NAMESPACE", "research-papers"), alias="PINECONE_NAMESPACE")
    pinecone_cloud: str = Field(default_factory=lambda: os.getenv("PINECONE_CLOUD", "aws"), alias="PINECONE_CLOUD")
    pinecone_region: str = Field(default_factory=lambda: os.getenv("PINECONE_REGION", "us-east-1"), alias="PINECONE_REGION")

    # --- Neon Postgres Relational Database ---
    # Pooled connection string for async application queries via asyncpg
    neon_database_url: str = Field(
        default_factory=lambda: os.getenv(
            "NEON_DATABASE_URL",
            "postgresql+asyncpg://postgres:postgres@localhost:5432/aurelium?ssl=require",
        ),
        alias="NEON_DATABASE_URL",
        description="Pooled Neon connection URL with asyncpg and SSL required",
    )
    # Direct connection string for Alembic migrations (bypasses transaction pooler)
    neon_direct_url: str = Field(
        default_factory=lambda: os.getenv(
            "NEON_DIRECT_URL",
            "postgresql://postgres:postgres@localhost:5432/aurelium?sslmode=require",
        ),
        alias="NEON_DIRECT_URL",
        description="Direct Neon connection URL for schema DDL and migrations",
    )

    # --- Embedding & Dimensions ---
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "allenai/specter2"), alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "768")), alias="EMBEDDING_DIM")

    # --- Retrieval Parameters ---
    retrieval_top_k: int = Field(default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "20")), alias="RETRIEVAL_TOP_K")
    rerank_top_k: int = Field(default_factory=lambda: int(os.getenv("RERANK_TOP_K", "5")), alias="RERANK_TOP_K")
    similarity_threshold: float = Field(default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", "0.65")), alias="SIMILARITY_THRESHOLD")
    max_context_tokens: int = Field(default_factory=lambda: int(os.getenv("MAX_CONTEXT_TOKENS", "8192")), alias="MAX_CONTEXT_TOKENS")

    # --- App Environment & Logging ---
    app_env: str = Field(default_factory=lambda: os.getenv("APP_ENV", "development"), alias="APP_ENV")
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"), alias="LOG_LEVEL")


settings = Settings()
