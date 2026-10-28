from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from contracts.schemas.models import Chunk, Paper
from ai.ingestion.arxiv import ArxivClient
from ai.ingestion.chunker import PaperChunker
from ai.ingestion.cleaner import PaperCleaner
from ai.ingestion.deduplicator import Deduplicator
from ai.ingestion.embedder import ChunkEmbedder
from ai.ingestion.semantic_scholar import SemanticScholarClient
from ai.ingestion.storage import IngestionStorageManager

logger = logging.getLogger("aurelium.ai.ingestion.pipeline")


@dataclass
class IngestionStats:
    """Summary metrics of an ingestion execution."""
    raw_fetched: int = 0
    cleaned_valid: int = 0
    unique_deduplicated: int = 0
    chunks_created: int = 0
    chunks_embedded: int = 0
    papers_stored_neon: int = 0
    chunks_stored_neon: int = 0
    vectors_synced_pinecone: int = 0


class IngestionPipeline:
    """
    End-to-end ingestion pipeline executing:
    Fetch -> Clean -> Deduplicate -> Chunk -> Embed -> Store (Neon + Pinecone).
    """

    def __init__(
        self,
        arxiv_client: ArxivClient | None = None,
        s2_client: SemanticScholarClient | None = None,
        chunker: PaperChunker | None = None,
        embedder: ChunkEmbedder | None = None,
        storage: IngestionStorageManager | None = None,
    ):
        self.arxiv_client = arxiv_client or ArxivClient()
        self.s2_client = s2_client or SemanticScholarClient()
        self.chunker = chunker or PaperChunker()
        self.embedder = embedder or ChunkEmbedder()
        self.storage = storage or IngestionStorageManager()

    async def run(
        self,
        topic: str,
        limit: int = 100,
        sources: Sequence[str] = ("arxiv", "semanticscholar"),
        namespace: str | None = None,
    ) -> IngestionStats:
        """Runs the ingestion pipeline for a given topic."""
        stats = IngestionStats()
        raw_papers: list[Paper] = []

        # 1. Fetch from requested sources
        if "arxiv" in sources:
            try:
                arxiv_papers = await self.arxiv_client.fetch_papers(topic=topic, limit=limit)
                raw_papers.extend(arxiv_papers)
            except Exception as e:
                logger.error(f"Failed fetching from arXiv: {e}")

        if "semanticscholar" in sources:
            try:
                s2_papers = await self.s2_client.fetch_papers(topic=topic, limit=limit)
                raw_papers.extend(s2_papers)
            except Exception as e:
                logger.error(f"Failed fetching from Semantic Scholar: {e}")

        stats.raw_fetched = len(raw_papers)
        logger.info(f"Step 1 Complete: Fetched {stats.raw_fetched} raw papers.")

        # 2. Cleaning
        cleaned_papers = PaperCleaner.clean_papers(raw_papers)
        stats.cleaned_valid = len(cleaned_papers)
        logger.info(f"Step 2 Complete: Cleaned {stats.cleaned_valid} valid papers (dropped {stats.raw_fetched - stats.cleaned_valid}).")

        # 3. Deduplication across sources
        dedup = Deduplicator()
        unique_papers = dedup.deduplicate(cleaned_papers)
        stats.unique_deduplicated = len(unique_papers)
        logger.info(f"Step 3 Complete: Deduplicated down to {stats.unique_deduplicated} unique papers.")

        if not unique_papers:
            return stats

        # 4. Chunking
        all_chunks: list[Chunk] = []
        for p in unique_papers:
            p_chunks = await self.chunker.chunk_paper(p)
            all_chunks.extend(p_chunks)

        stats.chunks_created = len(all_chunks)
        logger.info(f"Step 4 Complete: Created {stats.chunks_created} chunks.")

        # 5. Embedding (skip already stored chunks for idempotency)
        paper_ids = [p.id for p in unique_papers]
        synced_chunk_ids = await self.storage.get_existing_synced_chunk_ids(paper_ids)
        chunks_to_embed = self.embedder.filter_unstored_chunks(all_chunks, synced_chunk_ids)

        if chunks_to_embed:
            embedded_chunks = self.embedder.embed_chunks(chunks_to_embed)
            stats.chunks_embedded = len(embedded_chunks)
        else:
            embedded_chunks = []
            logger.info("All chunks already embedded and synced; skipping embedding.")

        # 6. Storage: Neon upsert + Pinecone sync
        # For chunks that were already embedded previously, combine with newly embedded
        final_chunks_to_store = embedded_chunks if embedded_chunks else all_chunks

        try:
            p_neon, c_neon, v_pinecone = await self.storage.store_papers_and_chunks(
                papers=unique_papers,
                chunks=final_chunks_to_store,
                namespace=namespace,
            )
            stats.papers_stored_neon = p_neon
            stats.chunks_stored_neon = c_neon
            stats.vectors_synced_pinecone = v_pinecone
        except Exception as store_err:
            logger.error(f"Storage error occurred: {store_err}")
            # Even if Pinecone sync fails, Neon upserts succeeded and marked chunks as pending
            stats.papers_stored_neon = len(unique_papers)
            stats.chunks_stored_neon = len(final_chunks_to_store)
            raise

        logger.info(
            f"Step 6 Complete: Ingestion finished. Papers in Neon: {stats.papers_stored_neon}, "
            f"Chunks in Neon: {stats.chunks_stored_neon}, Vectors in Pinecone: {stats.vectors_synced_pinecone}"
        )
        return stats
