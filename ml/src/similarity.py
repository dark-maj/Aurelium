"""
Aurelium - Phase 3: Cosine Similarity Helpers
Cosine for two vectors, cosine matrices for normalized embeddings, and a fast top-k.
"""

import argparse
import json
import os
from typing import Tuple

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale each row (or a single vector) to length 1, as float32. Zero vectors stay zero."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, eps)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity of two vectors: (a . b) / (|a| * |b|).
    Divides by the norms, so a and b do not need to be normalized. Returns 0.0 for a zero vector.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(a @ b / denom)


def cosine_matrix(q: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    All query-document cosines at once, for L2-normalized rows: Q @ D.T.

    Why this equals cosine: cos(a, b) = (a . b) / (|a| * |b|). When every row has length 1,
    the denominator is 1 * 1, so cos(a, b) = a . b. Entry (i, j) of Q @ D.T is exactly
    q_i . d_j, so the whole matrix is every cosine, computed in one matrix multiply.
    Shapes: Q (n_queries, dim), D (n_docs, dim) -> (n_queries, n_docs).
    """
    return np.asarray(q) @ np.asarray(d).T


def top_k(scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Indices and scores of the k highest scores, best first.

    np.argpartition finds the k winners in O(n) without sorting everything; then only those
    k are sorted. Ties are broken by the lower index, so results are deterministic.
    """
    scores = np.asarray(scores)
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=scores.dtype)

    winners = np.argpartition(-scores, k - 1)[:k]
    order = np.lexsort((winners, -scores[winners]))  # sort by score desc, then index asc
    idx = winners[order]
    return idx, scores[idx]


def nearest_neighbors(emb: np.ndarray, query_idx: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    The k rows of emb most similar to row query_idx, excluding the query itself.

    emb must be L2-normalized. The query's own score is set to -inf so it can never win;
    without this, every paper's nearest neighbor would be itself with cosine 1.0.
    """
    scores = cosine_matrix(emb[query_idx : query_idx + 1], emb)[0].astype(np.float64)
    scores[query_idx] = -np.inf
    k = min(k, emb.shape[0] - 1)
    return top_k(scores, k)


def load_embeddings(clean_dir: str) -> Tuple[np.ndarray, list]:
    """Read embeddings.npy and embedding_ids.json from clean_dir (read-only) and L2-normalize."""
    emb = np.load(os.path.join(clean_dir, "embeddings.npy"))
    with open(os.path.join(clean_dir, "embedding_ids.json"), encoding="utf-8") as f:
        ids = json.load(f)
    if len(ids) != emb.shape[0]:
        raise ValueError(f"{len(ids)} ids but {emb.shape[0]} embedding rows")
    return l2_normalize(emb), ids


def main() -> None:
    """CLI entry point: print the nearest neighbors of the first few papers."""
    parser = argparse.ArgumentParser(description="Aurelium Cosine Similarity Check")
    parser.add_argument("--clean-dir", default="data/clean", help="Folder with embeddings.npy and embedding_ids.json")
    parser.add_argument("--k", type=int, default=3, help="Neighbors to show per paper")
    parser.add_argument("--n-queries", type=int, default=3, help="How many papers to query")
    parser.add_argument("--test", action="store_true", help="Use data/test/clean")
    args = parser.parse_args()

    if args.test:
        args.clean_dir = "data/test/clean"

    print(f"=== Aurelium Similarity {'(TEST MODE)' if args.test else ''} ===")
    emb, ids = load_embeddings(args.clean_dir)
    print(f"Loaded {emb.shape[0]} embeddings of dim {emb.shape[1]}\n")

    for q in range(min(args.n_queries, emb.shape[0])):
        idx, scores = nearest_neighbors(emb, q, args.k)
        print(f"Query {ids[q].rsplit('/', 1)[-1]}")
        for i, s in zip(idx, scores):
            print(f"  {s:.3f}  {ids[i].rsplit('/', 1)[-1]}")


if __name__ == "__main__":
    main()
