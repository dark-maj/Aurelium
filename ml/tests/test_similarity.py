"""
Aurelium - Phase 3: tests for src.similarity
Uses tiny hand-made vectors whose cosines can be checked on paper.
"""

import json

import numpy as np
import pytest

from src.similarity import (
    cosine,
    cosine_matrix,
    l2_normalize,
    load_embeddings,
    nearest_neighbors,
    top_k,
)

A = [1.0, 0.0]
B = [0.6, 0.8]
C = [0.0, 1.0]


def test_cosine_known_values():
    assert cosine(A, A) == pytest.approx(1.0)
    assert cosine(A, B) == pytest.approx(0.6)
    assert cosine(A, C) == pytest.approx(0.0)
    assert cosine(A, [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_ignores_length():
    # [2, 0] points the same way as [1, 0], so the cosine is unchanged
    assert cosine([2.0, 0.0], B) == pytest.approx(cosine(A, B))


def test_cosine_zero_vector_is_zero():
    assert cosine([0.0, 0.0], A) == 0.0


def test_l2_normalize_rows_have_length_one():
    x = l2_normalize(np.array([[3.0, 4.0], [0.0, 2.0]]))
    assert x.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), [1.0, 1.0], rtol=1e-6)
    np.testing.assert_allclose(x[0], [0.6, 0.8], rtol=1e-6)


def test_l2_normalize_keeps_zero_row_zero():
    x = l2_normalize(np.array([[0.0, 0.0]]))
    assert not np.isnan(x).any()
    np.testing.assert_array_equal(x, [[0.0, 0.0]])


def test_cosine_matrix_matches_pairwise_cosine():
    rng = np.random.default_rng(0)
    q = l2_normalize(rng.normal(size=(2, 5)))
    d = l2_normalize(rng.normal(size=(3, 5)))
    m = cosine_matrix(q, d)
    assert m.shape == (2, 3)
    for i in range(2):
        for j in range(3):
            assert m[i, j] == pytest.approx(cosine(q[i], d[j]), abs=1e-6)


def test_top_k_best_first():
    idx, scores = top_k(np.array([0.1, 0.9, 0.5, 0.7]), 2)
    assert idx.tolist() == [1, 3]
    np.testing.assert_allclose(scores, [0.9, 0.7])


def test_top_k_ties_go_to_lower_index():
    idx, _ = top_k(np.array([0.5, 0.9, 0.9, 0.5]), 3)
    assert idx.tolist() == [1, 2, 0]


def test_top_k_k_larger_than_n_and_zero():
    idx, _ = top_k(np.array([0.2, 0.8]), 10)
    assert idx.tolist() == [1, 0]
    idx, scores = top_k(np.array([0.2, 0.8]), 0)
    assert len(idx) == 0 and len(scores) == 0


def test_nearest_neighbors_excludes_query():
    emb = l2_normalize(np.array([A, B, C]))
    idx, scores = nearest_neighbors(emb, 0, 2)
    assert idx.tolist() == [1, 2]
    np.testing.assert_allclose(scores, [0.6, 0.0], atol=1e-6)


def test_nearest_neighbors_with_exact_duplicate():
    # rows 0 and 1 are identical; the query must still not return itself
    emb = l2_normalize(np.array([A, A, C]))
    idx, scores = nearest_neighbors(emb, 1, 1)
    assert idx.tolist() == [0]
    assert scores[0] == pytest.approx(1.0)


def test_nearest_neighbors_caps_k():
    emb = l2_normalize(np.array([A, B]))
    idx, _ = nearest_neighbors(emb, 0, 5)
    assert idx.tolist() == [1]


def test_load_embeddings_normalizes_and_checks_ids(tmp_path):
    np.save(tmp_path / "embeddings.npy", np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32))
    (tmp_path / "embedding_ids.json").write_text(json.dumps(["W1", "W2"]), encoding="utf-8")
    emb, ids = load_embeddings(str(tmp_path))
    assert ids == ["W1", "W2"]
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), [1.0, 1.0], rtol=1e-6)


def test_load_embeddings_mismatch_raises(tmp_path):
    np.save(tmp_path / "embeddings.npy", np.zeros((2, 2), dtype=np.float32))
    (tmp_path / "embedding_ids.json").write_text(json.dumps(["W1"]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_embeddings(str(tmp_path))
