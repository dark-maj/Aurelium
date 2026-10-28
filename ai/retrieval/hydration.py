from __future__ import annotations

import json
import logging
from typing import Any, Sequence

from contracts.schemas.models import Paper

try:
    from ai.db.session import engine as default_engine
except ImportError:
    default_engine = None  # type: ignore

from ai.retrieval.fusion import FusedCandidate
from ai.retrieval.models import ScoredChunk, StageScores

try:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncEngine
except ImportError:
    def text(query: str) -> Any:  # type: ignore
        return query

    AsyncEngine = Any  # type: ignore

logger = logging.getLogger("aurelium.ai.retrieval.hydration")


class ChunkHydrator:
    """
    Hydrates fused candidate chunk IDs from Neon Postgres, assembling complete
    chunk text, character offsets, attached Paper metadata, and multi-stage scores.
    """

    def __init__(self, db_engine: AsyncEngine | None = None):
        self.engine = db_engine or default_engine

    async def hydrate(self, candidates: Sequence[FusedCandidate]) -> list[ScoredChunk]:
        """
        Hydrates fused candidates from Neon Postgres, preserving candidate rank order.
        """
        if not candidates:
            return []

        chunk_ids = [c.chunk_id for c in candidates]
        cand_map = {c.chunk_id: c for c in candidates}

        sql = text("""
            SELECT
                c.chunk_id, c.paper_id, c.section, c.text, c.char_start, c.char_end,
                p.id AS p_id, p.title AS p_title, p.abstract AS p_abstract, p.authors AS p_authors,
                p.venue AS p_venue, p.year AS p_year, p.published_date AS p_published_date,
                p.doi AS p_doi, p.arxiv_id AS p_arxiv_id, p.semantic_scholar_id AS p_s2_id,
                p.citation_count AS p_citations, p.fields_of_study AS p_fields, p.url AS p_url,
                p.metadata AS p_metadata
            FROM chunks c
            LEFT JOIN papers p ON p.id = c.paper_id
            WHERE c.chunk_id = ANY(:chunk_ids);
        """)

        hydrated_map: dict[str, ScoredChunk] = {}
        try:
            async with self.engine.connect() as conn:
                res = await conn.execute(sql, {"chunk_ids": chunk_ids})
                for row in res.fetchall():
                    c_id = row[0]
                    cand = cand_map.get(c_id)
                    if not cand:
                        continue

                    paper_obj = None
                    if row[6] is not None:
                        # Construct Paper model
                        authors_val = row[9] if isinstance(row[9], list) else (json.loads(row[9]) if row[9] else [])
                        fields_val = row[17] if isinstance(row[17], list) else (json.loads(row[17]) if row[17] else [])
                        meta_val = row[19] if isinstance(row[19], dict) else (json.loads(row[19]) if row[19] else {})

                        paper_obj = Paper(
                            id=row[6],
                            title=row[7] or "",
                            abstract=row[8] or "",
                            authors=authors_val,
                            venue=row[10],
                            year=row[11],
                            published_date=row[12],
                            doi=row[13],
                            arxiv_id=row[14],
                            semantic_scholar_id=row[15],
                            citation_count=int(row[16] or 0),
                            fields_of_study=fields_val,
                            url=row[18],
                            metadata=meta_val,
                        )

                    stage_scores = StageScores(
                        dense_score=cand.dense_score,
                        sparse_score=cand.sparse_score,
                        rrf_score=cand.rrf_score,
                    )

                    scored_chunk = ScoredChunk(
                        chunk_id=c_id,
                        paper_id=row[1],
                        section=row[2],
                        text=row[3],
                        char_start=int(row[4]),
                        char_end=int(row[5]),
                        paper=paper_obj,
                        scores=stage_scores,
                        final_score=cand.rrf_score,
                    )
                    hydrated_map[c_id] = scored_chunk

        except Exception as exc:
            logger.error(f"Error hydrating chunks from Neon: {exc}")

        # Assemble in original fused rank order
        ordered_results: list[ScoredChunk] = []
        for cand in candidates:
            if cand.chunk_id in hydrated_map:
                ordered_results.append(hydrated_map[cand.chunk_id])

        logger.debug(f"Hydrated {len(ordered_results)} of {len(candidates)} candidate chunks from Neon.")
        return ordered_results
