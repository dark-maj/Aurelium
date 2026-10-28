from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from contracts.interfaces.protocols import RankerProtocol
from contracts.interfaces.stubs import StubRanker
from ai.config import settings
from ai.retrieval.dense import DenseRetriever
from ai.retrieval.fusion import reciprocal_rank_fusion
from ai.retrieval.hydration import ChunkHydrator
from ai.retrieval.models import RetrievalFilters, RetrievalResult, ScoredChunk
from ai.retrieval.reranker import CrossEncoderReranker, apply_ranker_protocol
from ai.retrieval.sparse import SparseRetriever

logger = logging.getLogger("aurelium.ai.retrieval.service")


class HybridRetriever:
    """
    Hybrid Search Engine coordinating:
    1. Concurrent Dense (Pinecone) and Sparse (Neon TSVECTOR) search.
    2. Reciprocal Rank Fusion (RRF).
    3. Full chunk & paper hydration from Neon.
    4. Optional Cross-Encoder reranking.
    5. RankerProtocol reordering hook from contracts.
    6. Diagnostic weak-retrieval signal detection.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseRetriever | None = None,
        hydrator: ChunkHydrator | None = None,
        ranker_protocol: RankerProtocol | None = None,
        cross_encoder: CrossEncoderReranker | None = None,
        enable_cross_encoder: bool = False,
        k_rrf: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        candidate_pool_size: int = 40,
        similarity_threshold: float | None = None,
    ):
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.sparse_retriever = sparse_retriever or SparseRetriever()
        self.hydrator = hydrator or ChunkHydrator()
        self.ranker_protocol = ranker_protocol or StubRanker()
        self.cross_encoder = cross_encoder or CrossEncoderReranker()
        self.enable_cross_encoder = enable_cross_encoder
        self.k_rrf = k_rrf
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.candidate_pool_size = candidate_pool_size
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.similarity_threshold
        )

    async def retrieve(
        self,
        query: str,
        k: int = 10,
        filters: RetrievalFilters | None = None,
        enable_cross_encoder: bool | None = None,
        ranker: RankerProtocol | None = None,
        namespace: str | None = None,
    ) -> RetrievalResult:
        """
        Executes hybrid retrieval across Pinecone and Neon.
        Runs dense and sparse searches concurrently and tracks latency per stage.
        """
        latencies: dict[str, float] = {}
        t_start = time.perf_counter()

        pool_k = max(k * 2, self.candidate_pool_size)

        # 1. Run Dense and Sparse searches CONCURRENTLY
        t_search_start = time.perf_counter()

        async def _run_dense():
            t0 = time.perf_counter()
            res = await self.dense_retriever.search(
                query=query, top_k=pool_k, filters=filters, namespace=namespace
            )
            latencies["dense_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            return res

        async def _run_sparse():
            t0 = time.perf_counter()
            res = await self.sparse_retriever.search(
                query=query, top_k=pool_k, filters=filters
            )
            latencies["sparse_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            return res

        dense_matches, sparse_matches = await asyncio.gather(_run_dense(), _run_sparse())
        latencies["concurrent_search_latency_ms"] = round((time.perf_counter() - t_search_start) * 1000, 2)

        # 2. Reciprocal Rank Fusion
        t_fusion = time.perf_counter()
        fused_candidates = reciprocal_rank_fusion(
            dense_results=dense_matches,
            sparse_results=sparse_matches,
            k_rrf=self.k_rrf,
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
            top_k=pool_k,
        )
        latencies["fusion_latency_ms"] = round((time.perf_counter() - t_fusion) * 1000, 2)

        # 3. Hydrate chunks from Neon
        t_hydrate = time.perf_counter()
        hydrated_chunks = await self.hydrator.hydrate(fused_candidates)
        latencies["hydration_latency_ms"] = round((time.perf_counter() - t_hydrate) * 1000, 2)

        # 4. Optional Cross-Encoder Rerank (off by default)
        use_ce = self.enable_cross_encoder if enable_cross_encoder is None else enable_cross_encoder
        if use_ce and hydrated_chunks:
            t_ce = time.perf_counter()
            hydrated_chunks = self.cross_encoder.rerank(query=query, chunks=hydrated_chunks)
            latencies["cross_encoder_latency_ms"] = round((time.perf_counter() - t_ce) * 1000, 2)

        # 5. Hook: Pass candidates through RankerProtocol from contracts
        t_ranker = time.perf_counter()
        active_ranker = ranker or self.ranker_protocol
        final_ranked_chunks = apply_ranker_protocol(
            query=query, chunks=hydrated_chunks, ranker=active_ranker
        )
        latencies["ranker_hook_latency_ms"] = round((time.perf_counter() - t_ranker) * 1000, 2)

        # 6. Weak-Retrieval Diagnostic Signal
        effective_threshold = (
            filters.similarity_threshold
            if filters and filters.similarity_threshold is not None
            else self.similarity_threshold
        )

        is_weak = False
        weak_reason = None

        if not final_ranked_chunks:
            is_weak = True
            weak_reason = "No candidate chunks found matching the query and filters."
        else:
            top_chunk = final_ranked_chunks[0]
            # Check if top candidate dense score or ranker score is below threshold
            top_dense = top_chunk.scores.dense_score
            top_final = top_chunk.final_score

            # Evaluation heuristic: if dense similarity is available and below threshold
            if top_dense is not None and top_dense < effective_threshold:
                is_weak = True
                weak_reason = (
                    f"Top dense similarity score ({top_dense:.3f}) falls below "
                    f"threshold ({effective_threshold:.3f}); literature coverage is weak."
                )
            elif top_final < 0.10 and (top_dense is None or top_dense < effective_threshold):
                is_weak = True
                weak_reason = (
                    f"Top final relevance score ({top_final:.3f}) falls below "
                    f"confidence threshold; literature coverage is weak."
                )

        total_latency = round((time.perf_counter() - t_start) * 1000, 2)
        latencies["total_latency_ms"] = total_latency

        logger.info(
            f"Retrieval complete for query '{query[:30]}...'",
            extra={
                "returned_chunks": len(final_ranked_chunks[:k]),
                "is_weak": is_weak,
                "latencies": latencies,
            },
        )

        return RetrievalResult(
            query=query,
            chunks=final_ranked_chunks[:k],
            is_weak=is_weak,
            weak_reason=weak_reason,
            latencies_ms=latencies,
        )


_default_retriever: HybridRetriever | None = None


def get_default_retriever() -> HybridRetriever:
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = HybridRetriever()
    return _default_retriever


async def retrieve(
    query: str,
    k: int = 10,
    filters: RetrievalFilters | dict[str, Any] | None = None,
    *,
    enable_cross_encoder: bool = False,
    ranker: RankerProtocol | None = None,
    namespace: str | None = None,
    retriever_instance: HybridRetriever | None = None,
) -> RetrievalResult:
    """
    Primary interface function for the Aurelium hybrid retrieval module.

    Executes concurrent dense (Pinecone) and sparse (Neon full-text) search,
    reciprocal rank fusion, Neon chunk hydration, optional cross-encoder reranking,
    and RankerProtocol reordering.

    Args:
        query: User search query or question.
        k: Maximum number of hydrated chunks to return.
        filters: Optional filters (year_min, year_max, fields_of_study, similarity_threshold).
        enable_cross_encoder: Flag to run cross-encoder scoring (default: False).
        ranker: Optional RankerProtocol hook instance (defaults to StubRanker).
        namespace: Target Pinecone vector namespace.
        retriever_instance: Optional injected HybridRetriever instance.

    Returns:
        RetrievalResult: Hydrated chunks, weak-retrieval flag, and stage latency metrics.
    """
    if isinstance(filters, dict):
        parsed_filters = RetrievalFilters(**filters)
    else:
        parsed_filters = filters

    active_engine = retriever_instance or get_default_retriever()
    return await active_engine.retrieve(
        query=query,
        k=k,
        filters=parsed_filters,
        enable_cross_encoder=enable_cross_encoder,
        ranker=ranker,
        namespace=namespace,
    )
