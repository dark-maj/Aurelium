from typing import List, Set
import math


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Computes Precision@k."""
    if k <= 0 or not retrieved_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Computes Recall@k."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def mrr_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Computes Mean Reciprocal Rank at k."""
    for rank_idx, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank_idx
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Computes Normalized Discounted Cumulative Gain at k (binary relevance)."""
    dcg = 0.0
    for rank_idx, doc_id in enumerate(retrieved_ids[:k], start=1):
        rel = 1.0 if doc_id in relevant_ids else 0.0
        dcg += rel / math.log2(rank_idx + 1)

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    return (dcg / idcg) if idcg > 0.0 else 0.0
