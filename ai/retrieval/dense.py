from __future__ import annotations

import logging
from typing import Any
from ai.config import settings
from ai.ingestion.embedder import ChunkEmbedder
from ai.retrieval.models import RetrievalFilters
from ai.retrieval.pinecone_client import PineconeVectorClient, get_pinecone_client

logger = logging.getLogger("aurelium.ai.retrieval.dense")


def build_pinecone_filter(filters: RetrievalFilters | None) -> dict[str, Any] | None:
    """Constructs Pinecone metadata filter clauses from RetrievalFilters."""
    if not filters:
        return None

    clauses: list[dict[str, Any]] = []

    # Year range filter
    if filters.year_min is not None and filters.year_max is not None:
        clauses.append({"year": {"$gte": filters.year_min, "$lte": filters.year_max}})
    elif filters.year_min is not None:
        clauses.append({"year": {"$gte": filters.year_min}})
    elif filters.year_max is not None:
        clauses.append({"year": {"$lte": filters.year_max}})

    # Fields of study filter
    if filters.fields_of_study:
        clauses.append({"fields_of_study": {"$in": filters.fields_of_study}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


class DenseRetriever:
    """Performs dense vector retrieval against Pinecone using local query embedding."""

    def __init__(
        self,
        pinecone_client: PineconeVectorClient | None = None,
        embedder: ChunkEmbedder | None = None,
    ):
        self.pinecone = pinecone_client or get_pinecone_client()
        self.embedder = embedder or ChunkEmbedder()

    async def search(
        self,
        query: str,
        top_k: int = 50,
        filters: RetrievalFilters | None = None,
        namespace: str | None = None,
    ) -> list[tuple[str, float]]:
        """
        Embeds query locally and queries Pinecone for nearest chunks.
        Returns list of (chunk_id, dense_score) sorted descending.
        """
        ns = namespace or settings.pinecone_namespace
        filter_dict = build_pinecone_filter(filters)

        # 1. Embed query
        from contracts.schemas.models import Chunk
        dummy_chunk = Chunk(
            chunk_id="_query",
            paper_id="_query",
            text=query,
            char_start=0,
            char_end=len(query),
        )
        embedded_query = self.embedder.embed_chunks([dummy_chunk])[0]
        query_vec = embedded_query.embedding
        if not query_vec:
            logger.error("Failed to generate embedding vector for query")
            return []

        # 2. Query Pinecone
        matches = await self.pinecone.query(
            vector=query_vec,
            top_k=top_k,
            namespace=ns,
            filter_dict=filter_dict,
        )

        results = [(m.chunk_id, m.score) for m in matches if m.chunk_id]
        logger.debug(f"Dense search returned {len(results)} matches for query '{query[:30]}...'")
        return results
