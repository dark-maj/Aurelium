from __future__ import annotations

import logging
from typing import Any

from ai.retrieval.models import RetrievalFilters

try:
    from ai.db.session import engine as default_engine
except ImportError:
    default_engine = None  # type: ignore

try:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncEngine
except ImportError:
    def text(query: str) -> Any:  # type: ignore
        return query

    AsyncEngine = Any  # type: ignore

logger = logging.getLogger("aurelium.ai.retrieval.sparse")


class SparseRetriever:
    """
    Performs sparse BM25-style keyword search over Neon Postgres using the
    pre-indexed chunks.tsv TSVECTOR and PostgreSQL ts_rank.
    Applies the same metadata filters (year range, fields of study).
    """

    def __init__(self, db_engine: AsyncEngine | None = None):
        self.engine = db_engine or default_engine

    async def search(
        self,
        query: str,
        top_k: int = 50,
        filters: RetrievalFilters | None = None,
    ) -> list[tuple[str, float]]:
        """
        Executes full-text search against Neon Postgres.
        Returns list of (chunk_id, sparse_score) sorted descending.
        """
        if not query or not query.strip():
            return []

        year_min = filters.year_min if filters else None
        year_max = filters.year_max if filters else None
        fields = filters.fields_of_study if filters else None

        sql = text("""
            SELECT c.chunk_id, ts_rank_cd(c.tsv, q) AS rank_score
            FROM chunks c
            JOIN papers p ON p.id = c.paper_id,
            plainto_tsquery('english', :query_text) q
            WHERE c.tsv @@ q
              AND (:year_min IS NULL OR p.year >= :year_min)
              AND (:year_max IS NULL OR p.year <= :year_max)
              AND (
                  :has_fields = FALSE
                  OR EXISTS (
                      SELECT 1 FROM jsonb_array_elements_text(p.fields_of_study) f
                      WHERE f = ANY(:fields_list)
                  )
              )
            ORDER BY rank_score DESC
            LIMIT :limit;
        """)

        params = {
            "query_text": query,
            "year_min": year_min,
            "year_max": year_max,
            "has_fields": bool(fields),
            "fields_list": list(fields) if fields else [],
            "limit": top_k,
        }

        try:
            async with self.engine.connect() as conn:
                res = await conn.execute(sql, params)
                rows = res.fetchall()
                results = [(row[0], float(row[1])) for row in rows]
                logger.debug(f"Sparse search returned {len(results)} matches for query '{query[:30]}...'")
                return results
        except Exception as exc:
            logger.error(f"Error during sparse Postgres search: {exc}")
            return []
