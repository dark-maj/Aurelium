from __future__ import annotations

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from ai.config import settings

# Asyncpg serverless parameters:
# - statement_cache_size=0: required for Neon PgBouncer transaction pooling mode
# - ssl="require": enforces TLS for remote Neon endpoint
_CONNECT_ARGS = {
    "ssl": "require",
    "statement_cache_size": 0,
}


def create_neon_engine(url: str | None = None) -> AsyncEngine:
    """
    Creates an async SQLAlchemy engine optimized for Neon serverless Postgres.
    Pool settings handle auto-suspend, cold starts, and PgBouncer connection drops.
    """
    db_url = url or settings.neon_database_url
    return create_async_engine(
        db_url,
        connect_args=_CONNECT_ARGS,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,       # Verifies connection liveness before checkout
        pool_recycle=300,        # Recycles connections every 5 min to avoid stale sockets
        pool_timeout=30.0,
        echo=False,
    )


engine: AsyncEngine = create_neon_engine()
async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency provider for FastAPI routes and LangGraph agents."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
