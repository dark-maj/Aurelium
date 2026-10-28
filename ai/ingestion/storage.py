from __future__ import annotations

import json
import logging
from typing import Any, Sequence

try:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncEngine
except ImportError:
    def text(query: str) -> Any:  # type: ignore
        return query

    AsyncEngine = Any  # type: ignore

from contracts.schemas.models import Chunk, Paper
from ai.config import settings

try:
    from ai.db.session import engine as default_engine
except ImportError:
    default_engine = None  # type: ignore

from ai.retrieval.pinecone_client import PineconeVectorClient, VectorRecord, get_pinecone_client

logger = logging.getLogger("aurelium.ai.ingestion.storage")


class IngestionStorageManager:
    """
    Manages dual-storage writes to Neon Postgres and Pinecone vector store.
    Guarantees:
    - Idempotent upserts for papers and chunks.
    - Full-text search tsvector synchronization in Neon.
    - Metadata enforcement in Pinecone.
    - Partial-failure recovery: if Pinecone write fails after Neon write,
      the chunk is recorded with metadata 'pinecone_status': 'pending'.
    """

    def __init__(
        self,
        db_engine: AsyncEngine | None = None,
        pinecone_client: PineconeVectorClient | None = None,
    ):
        self.engine = db_engine or default_engine
        self.pinecone = pinecone_client or get_pinecone_client()

    async def get_existing_synced_chunk_ids(self, paper_ids: Sequence[str]) -> set[str]:
        """Returns chunk IDs that have already been stored in Neon and synced to Pinecone."""
        if not paper_ids:
            return set()

        query = text("""
            SELECT chunk_id FROM chunks
            WHERE paper_id = ANY(:paper_ids)
              AND (metadata->>'pinecone_status' = 'synced')
        """)
        try:
            async with self.engine.connect() as conn:
                result = await conn.execute(query, {"paper_ids": list(paper_ids)})
                return {row[0] for row in result.fetchall()}
        except Exception as exc:
            logger.debug(f"Could not query existing chunk IDs from DB: {exc}")
            return set()

    async def upsert_papers_neon(self, papers: Sequence[Paper]) -> int:
        """Idempotently upserts papers to Neon Postgres."""
        if not papers:
            return 0

        upsert_stmt = text("""
            INSERT INTO papers (
                id, title, abstract, authors, venue, year, published_date,
                doi, arxiv_id, semantic_scholar_id, citation_count,
                fields_of_study, url, metadata
            ) VALUES (
                :id, :title, :abstract, CAST(:authors AS jsonb), :venue, :year, :published_date,
                :doi, :arxiv_id, :semantic_scholar_id, :citation_count,
                CAST(:fields_of_study AS jsonb), :url, CAST(:metadata AS jsonb)
            )
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title,
                abstract = EXCLUDED.abstract,
                authors = EXCLUDED.authors,
                venue = EXCLUDED.venue,
                year = EXCLUDED.year,
                published_date = EXCLUDED.published_date,
                doi = COALESCE(EXCLUDED.doi, papers.doi),
                arxiv_id = COALESCE(EXCLUDED.arxiv_id, papers.arxiv_id),
                semantic_scholar_id = COALESCE(EXCLUDED.semantic_scholar_id, papers.semantic_scholar_id),
                citation_count = GREATEST(EXCLUDED.citation_count, papers.citation_count),
                fields_of_study = EXCLUDED.fields_of_study,
                url = COALESCE(EXCLUDED.url, papers.url),
                metadata = papers.metadata || EXCLUDED.metadata;
        """)

        params = [
            {
                "id": p.id,
                "title": p.title,
                "abstract": p.abstract,
                "authors": json.dumps(p.authors),
                "venue": p.venue,
                "year": p.year,
                "published_date": p.published_date,
                "doi": p.doi,
                "arxiv_id": p.arxiv_id,
                "semantic_scholar_id": p.semantic_scholar_id,
                "citation_count": p.citation_count,
                "fields_of_study": json.dumps(p.fields_of_study),
                "url": p.url,
                "metadata": json.dumps(p.metadata),
            }
            for p in papers
        ]

        async with self.engine.begin() as conn:
            await conn.execute(upsert_stmt, params)

        logger.info(f"Upserted {len(papers)} papers into Neon.")
        return len(papers)

    async def upsert_chunks_neon(self, chunks: Sequence[Chunk], pinecone_status: str = "pending") -> int:
        """
        Idempotently upserts chunks into Neon with initial pinecone_status.
        TSVECTOR is automatically populated by Postgres persisted computation.
        """
        if not chunks:
            return 0

        upsert_stmt = text("""
            INSERT INTO chunks (
                chunk_id, paper_id, section, text, char_start, char_end,
                token_count, metadata
            ) VALUES (
                :chunk_id, :paper_id, :section, :text, :char_start, :char_end,
                :token_count, CAST(:metadata AS jsonb)
            )
            ON CONFLICT (chunk_id) DO UPDATE SET
                section = EXCLUDED.section,
                text = EXCLUDED.text,
                char_start = EXCLUDED.char_start,
                char_end = EXCLUDED.char_end,
                token_count = EXCLUDED.token_count,
                metadata = chunks.metadata || EXCLUDED.metadata;
        """)

        params = []
        for c in chunks:
            meta = dict(c.metadata)
            meta["pinecone_status"] = pinecone_status
            params.append({
                "chunk_id": c.chunk_id,
                "paper_id": c.paper_id,
                "section": c.section,
                "text": c.text,
                "char_start": c.char_start,
                "char_end": c.char_end,
                "token_count": c.token_count,
                "metadata": json.dumps(meta),
            })

        async with self.engine.begin() as conn:
            await conn.execute(upsert_stmt, params)

        logger.info(f"Upserted {len(chunks)} chunks into Neon (status: {pinecone_status}).")
        return len(chunks)

    async def update_chunks_pinecone_status(self, chunk_ids: Sequence[str], status: str) -> None:
        """Updates the pinecone_status in chunk metadata in Neon."""
        if not chunk_ids:
            return

        update_stmt = text("""
            UPDATE chunks
            SET metadata = jsonb_set(metadata, '{pinecone_status}', to_jsonb(:status::text))
            WHERE chunk_id = ANY(:chunk_ids);
        """)
        async with self.engine.begin() as conn:
            await conn.execute(update_stmt, {"chunk_ids": list(chunk_ids), "status": status})

    async def sync_to_pinecone(
        self,
        chunks: Sequence[Chunk],
        namespace: str | None = None,
        batch_size: int = 100,
    ) -> int:
        """
        Pushes chunk vectors and metadata into Pinecone.
        Updates Neon chunk status to 'synced' upon success.
        If Pinecone fails, leaves status as 'pending' for retry.
        """
        chunks_with_vectors = [c for c in chunks if c.embedding]
        if not chunks_with_vectors:
            return 0

        ns = namespace or settings.pinecone_namespace
        records = [
            VectorRecord(
                id=c.chunk_id,
                values=c.embedding,  # type: ignore
                paper_id=c.paper_id,
                chunk_id=c.chunk_id,
                year=c.metadata.get("year"),
                fields_of_study=c.metadata.get("fields_of_study", []),
                metadata=dict(c.metadata),
            )
            for c in chunks_with_vectors
        ]

        total_synced = 0
        for i in range(0, len(records), batch_size):
            batch_records = records[i : i + batch_size]
            batch_chunk_ids = [r.chunk_id for r in batch_records]

            try:
                upserted = await self.pinecone.upsert_records(batch_records, namespace=ns)
                total_synced += upserted
                # Mark as synced in Neon
                await self.update_chunks_pinecone_status(batch_chunk_ids, "synced")
            except Exception as exc:
                logger.error(
                    f"Pinecone batch upsert failed! Chunks remain recorded as 'pending' in Neon. Error: {exc}",
                    extra={"failed_chunk_count": len(batch_chunk_ids)},
                )
                # Keep as pending and re-raise so pipeline is aware of partial failure
                raise

        return total_synced

    async def store_papers_and_chunks(
        self,
        papers: Sequence[Paper],
        chunks: Sequence[Chunk],
        namespace: str | None = None,
    ) -> tuple[int, int, int]:
        """
        High-level atomic pipeline:
        1. Upsert papers to Neon.
        2. Upsert chunks to Neon with 'pinecone_status': 'pending'.
        3. Sync vectors to Pinecone.
        4. On Pinecone success, mark status as 'synced'.
        Returns (papers_count, chunks_count, pinecone_synced_count).
        """
        # Step 1 & 2: Neon writes
        p_count = await self.upsert_papers_neon(papers)
        c_count = await self.upsert_chunks_neon(chunks, pinecone_status="pending")

        # Step 3 & 4: Pinecone write with status update
        synced_count = await self.sync_to_pinecone(chunks, namespace=namespace)

        return (p_count, c_count, synced_count)

    async def retry_pending_chunks(
        self,
        embedder: Any,
        namespace: str | None = None,
        limit: int = 200,
    ) -> int:
        """
        Queries chunks that failed Pinecone sync on prior runs and retries them.
        """
        query = text("""
            SELECT chunk_id, paper_id, section, text, char_start, char_end, token_count, metadata
            FROM chunks
            WHERE metadata->>'pinecone_status' = 'pending'
            LIMIT :limit;
        """)

        pending_chunks: list[Chunk] = []
        async with self.engine.connect() as conn:
            result = await conn.execute(query, {"limit": limit})
            for row in result.fetchall():
                pending_chunks.append(
                    Chunk(
                        chunk_id=row[0],
                        paper_id=row[1],
                        section=row[2],
                        text=row[3],
                        char_start=row[4],
                        char_end=row[5],
                        token_count=row[6],
                        metadata=row[7] if isinstance(row[7], dict) else json.loads(row[7]),
                    )
                )

        if not pending_chunks:
            logger.info("No pending chunks found for Pinecone retry.")
            return 0

        logger.info(f"Retrying {len(pending_chunks)} pending chunks for Pinecone sync.")
        embedded_chunks = embedder.embed_chunks(pending_chunks)
        return await self.sync_to_pinecone(embedded_chunks, namespace=namespace)
