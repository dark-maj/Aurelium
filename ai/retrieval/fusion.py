from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class FusedCandidate:
    """Intermediate candidate output from reciprocal rank fusion."""
    chunk_id: str
    rrf_score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None


def reciprocal_rank_fusion(
    dense_results: Sequence[tuple[str, float]],
    sparse_results: Sequence[tuple[str, float]],
    k_rrf: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    top_k: int = 20,
) -> list[FusedCandidate]:
    """
    Applies Reciprocal Rank Fusion (RRF) over dense and sparse ranked lists:
        RRF_score(d) = dense_weight * (1 / (k_rrf + rank_dense)) + sparse_weight * (1 / (k_rrf + rank_sparse))

    Args:
        dense_results: Ranked sequence of (chunk_id, dense_score) from Pinecone.
        sparse_results: Ranked sequence of (chunk_id, sparse_score) from Postgres ts_rank.
        k_rrf: Ranking smoothing constant (default: 60).
        dense_weight: Multiplier weight for dense vector rankings.
        sparse_weight: Multiplier weight for sparse keyword rankings.
        top_k: Number of fused candidates to return.

    Returns:
        Sorted list of FusedCandidate instances ordered by rrf_score descending.
    """
    scores: dict[str, float] = {}
    dense_data: dict[str, tuple[float, int]] = {}
    sparse_data: dict[str, tuple[float, int]] = {}

    # 1. Process dense rankings (1-indexed rank)
    for rank, (chunk_id, dense_score) in enumerate(dense_results, start=1):
        dense_data[chunk_id] = (dense_score, rank)
        rrf_contrib = dense_weight * (1.0 / (k_rrf + rank))
        scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_contrib

    # 2. Process sparse rankings (1-indexed rank)
    for rank, (chunk_id, sparse_score) in enumerate(sparse_results, start=1):
        sparse_data[chunk_id] = (sparse_score, rank)
        rrf_contrib = sparse_weight * (1.0 / (k_rrf + rank))
        scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_contrib

    # 3. Build fused candidates
    fused: list[FusedCandidate] = []
    for chunk_id, rrf_score in scores.items():
        d_val, d_rank = dense_data.get(chunk_id, (None, None))
        s_val, s_rank = sparse_data.get(chunk_id, (None, None))
        fused.append(
            FusedCandidate(
                chunk_id=chunk_id,
                rrf_score=round(rrf_score, 6),
                dense_score=d_val,
                sparse_score=s_val,
                dense_rank=d_rank,
                sparse_rank=s_rank,
            )
        )

    # Sort descending by rrf_score
    fused.sort(key=lambda x: x.rrf_score, reverse=True)
    return fused[:top_k]
