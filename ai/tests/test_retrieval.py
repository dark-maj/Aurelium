from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock

from contracts.interfaces.protocols import RankerProtocol
from contracts.interfaces.stubs import StubRanker
from contracts.schemas.models import Paper
from ai.retrieval.dense import DenseRetriever, build_pinecone_filter
from ai.retrieval.fusion import FusedCandidate, reciprocal_rank_fusion
from ai.retrieval.hydration import ChunkHydrator
from ai.retrieval.models import (
    RetrievalFilters,
    RetrievalResult,
    ScoredChunk,
    StageScores,
)
from ai.retrieval.reranker import CrossEncoderReranker, apply_ranker_protocol
from ai.retrieval.service import HybridRetriever, retrieve
from ai.retrieval.sparse import SparseRetriever


class TestFusionMath(unittest.TestCase):
    """Verifies reciprocal rank fusion formula, weights, and overlapping document boosts."""

    def test_rrf_scoring_and_weights(self):
        dense_results = [("chunk_a", 0.95), ("chunk_b", 0.85), ("chunk_c", 0.75)]
        sparse_results = [("chunk_b", 4.5), ("chunk_a", 3.2), ("chunk_d", 2.1)]

        # k_rrf = 60, dense_weight = 1.0, sparse_weight = 1.0
        # chunk_a: rank_d=1, rank_s=2 -> 1/61 + 1/62 = 0.016393 + 0.016129 = 0.032522
        # chunk_b: rank_d=2, rank_s=1 -> 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
        # chunk_c: rank_d=3, rank_s=None -> 1/63 = 0.015873
        # chunk_d: rank_d=None, rank_s=3 -> 1/63 = 0.015873
        fused = reciprocal_rank_fusion(
            dense_results, sparse_results, k_rrf=60, dense_weight=1.0, sparse_weight=1.0
        )

        self.assertEqual(len(fused), 4)
        top_ids = {fused[0].chunk_id, fused[1].chunk_id}
        self.assertEqual(top_ids, {"chunk_a", "chunk_b"})
        self.assertAlmostEqual(fused[0].rrf_score, 0.032522, places=4)

        # Verify custom dense weight boost
        fused_weighted = reciprocal_rank_fusion(
            dense_results, sparse_results, k_rrf=60, dense_weight=2.0, sparse_weight=1.0
        )
        # chunk_a: 2*(1/61) + 1/62 = 0.032786 + 0.016129 = 0.048915
        # chunk_b: 2*(1/62) + 1/61 = 0.032258 + 0.016393 = 0.048651
        self.assertEqual(fused_weighted[0].chunk_id, "chunk_a")
        self.assertGreater(fused_weighted[0].rrf_score, fused_weighted[1].rrf_score)


class TestFilters(unittest.TestCase):
    """Verifies that retrieval filters correctly format Pinecone and SQL queries."""

    def test_pinecone_filter_building(self):
        f1 = RetrievalFilters(year_min=2020, year_max=2024, fields_of_study=["CS", "AI"])
        p_filter = build_pinecone_filter(f1)
        self.assertIn("$and", p_filter)
        self.assertEqual(p_filter["$and"][0], {"year": {"$gte": 2020, "$lte": 2024}})
        self.assertEqual(p_filter["$and"][1], {"fields_of_study": {"$in": ["CS", "AI"]}})

        f2 = RetrievalFilters(year_min=2023)
        self.assertEqual(build_pinecone_filter(f2), {"year": {"$gte": 2023}})


class TestHydration(unittest.IsolatedAsyncioTestCase):
    """Verifies that ChunkHydrator loads all chunk text, character offsets, attached paper, and stage scores."""

    async def test_hydration_from_mock_neon(self):
        mock_engine = MagicMock()
        mock_conn = AsyncMock()
        mock_engine.connect.return_value.__aenter__.return_value = mock_conn

        # Mock DB row
        # (chunk_id, paper_id, section, text, char_start, char_end,
        #  p_id, p_title, p_abstract, p_authors, p_venue, p_year, p_published_date,
        #  p_doi, p_arxiv_id, p_s2_id, p_citations, p_fields, p_url, p_metadata)
        mock_row = (
            "chunk_001", "paper_001", "Abstract", "This is the abstract chunk text.", 0, 32,
            "paper_001", "DPO Paper", "Full abstract", ["Author A"], "NeurIPS", 2023, None,
            "10.1234/test", "2305.18290", "s2_123", 150, ["Computer Science"], "https://arxiv.org",
            {"source": "arxiv"}
        )
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]
        mock_conn.execute.return_value = mock_result

        hydrator = ChunkHydrator(db_engine=mock_engine)
        fused = [
            FusedCandidate(chunk_id="chunk_001", rrf_score=0.032, dense_score=0.88, sparse_score=2.5)
        ]

        hydrated = await hydrator.hydrate(fused)
        self.assertEqual(len(hydrated), 1)
        c = hydrated[0]
        self.assertEqual(c.chunk_id, "chunk_001")
        self.assertEqual(c.paper_id, "paper_001")
        self.assertEqual(c.section, "Abstract")
        self.assertEqual(c.text, "This is the abstract chunk text.")
        self.assertEqual(c.char_start, 0)
        self.assertEqual(c.char_end, 32)
        self.assertIsNotNone(c.paper)
        self.assertEqual(c.paper.title, "DPO Paper")
        self.assertEqual(c.scores.dense_score, 0.88)
        self.assertEqual(c.scores.sparse_score, 2.5)
        self.assertEqual(c.scores.rrf_score, 0.032)


class TestRankerHookAndReordering(unittest.TestCase):
    """Verifies that RankerProtocol reorders candidates and sets ranker_score."""

    def test_ranker_protocol_hook_reorders_chunks(self):
        p1 = Paper(id="p1", title="Alpha Paper", abstract="About transformers.")
        p2 = Paper(id="p2", title="Beta Paper", abstract="About reinforcement learning.")

        c1 = ScoredChunk(
            chunk_id="c1", paper_id="p1", text="Text 1", char_start=0, char_end=6,
            paper=p1, final_score=0.03
        )
        c2 = ScoredChunk(
            chunk_id="c2", paper_id="p2", text="Text 2", char_start=0, char_end=6,
            paper=p2, final_score=0.02
        )

        # Mock RankerProtocol returning higher score for p2 than p1
        mock_ranker = MagicMock(spec=RankerProtocol)
        mock_ranker.rank.return_value = [0.35, 0.92]  # p1 gets 0.35, p2 gets 0.92

        reordered = apply_ranker_protocol("rl query", [c1, c2], ranker=mock_ranker)
        self.assertEqual(len(reordered), 2)
        # c2 should now be first because 0.92 > 0.35
        self.assertEqual(reordered[0].chunk_id, "c2")
        self.assertEqual(reordered[0].scores.ranker_score, 0.92)
        self.assertEqual(reordered[1].chunk_id, "c1")
        self.assertEqual(reordered[1].scores.ranker_score, 0.35)


class TestWeakRetrievalFlag(unittest.IsolatedAsyncioTestCase):
    """Verifies weak-retrieval signal triggering when similarity scores fall below threshold."""

    async def test_weak_flag_triggered_on_low_scores(self):
        mock_dense = AsyncMock(spec=DenseRetriever)
        mock_dense.search.return_value = [("c1", 0.42)]  # 0.42 < threshold 0.65

        mock_sparse = AsyncMock(spec=SparseRetriever)
        mock_sparse.search.return_value = [("c1", 0.5)]

        mock_hydrator = AsyncMock(spec=ChunkHydrator)
        mock_hydrator.hydrate.return_value = [
            ScoredChunk(
                chunk_id="c1", paper_id="p1", text="Low score chunk", char_start=0, char_end=15,
                scores=StageScores(dense_score=0.42, sparse_score=0.5, rrf_score=0.016),
                final_score=0.016,
            )
        ]

        # Use stub ranker
        retriever = HybridRetriever(
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
            hydrator=mock_hydrator,
            similarity_threshold=0.65,
        )

        res = await retriever.retrieve("niche topic query", k=5)
        self.assertTrue(res.is_weak)
        self.assertIn("falls below threshold", res.weak_reason)

    async def test_weak_flag_false_on_confident_retrieval(self):
        mock_dense = AsyncMock(spec=DenseRetriever)
        mock_dense.search.return_value = [("c1", 0.88)]  # 0.88 >= 0.65

        mock_sparse = AsyncMock(spec=SparseRetriever)
        mock_sparse.search.return_value = [("c1", 3.2)]

        mock_hydrator = AsyncMock(spec=ChunkHydrator)
        mock_hydrator.hydrate.return_value = [
            ScoredChunk(
                chunk_id="c1", paper_id="p1", text="High score chunk", char_start=0, char_end=16,
                scores=StageScores(dense_score=0.88, sparse_score=3.2, rrf_score=0.032),
                final_score=0.032,
            )
        ]

        retriever = HybridRetriever(
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
            hydrator=mock_hydrator,
            similarity_threshold=0.65,
        )

        res = await retriever.retrieve("direct preference optimization", k=5)
        self.assertFalse(res.is_weak)
        self.assertIsNone(res.weak_reason)


class TestConcurrencyAndPublicFunction(unittest.IsolatedAsyncioTestCase):
    """Verifies concurrent execution, latency metrics, and public retrieve() interface."""

    async def test_concurrent_execution_and_latencies(self):
        mock_dense = AsyncMock(spec=DenseRetriever)
        mock_dense.search.return_value = [("c1", 0.85)]

        mock_sparse = AsyncMock(spec=SparseRetriever)
        mock_sparse.search.return_value = [("c1", 2.1)]

        mock_hydrator = AsyncMock(spec=ChunkHydrator)
        mock_hydrator.hydrate.return_value = [
            ScoredChunk(
                chunk_id="c1", paper_id="p1", text="Sample chunk", char_start=0, char_end=12,
                scores=StageScores(dense_score=0.85, sparse_score=2.1, rrf_score=0.032),
                final_score=0.85,
            )
        ]

        custom_engine = HybridRetriever(
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
            hydrator=mock_hydrator,
            similarity_threshold=0.60,
        )

        # Call public retrieve() function
        res = await retrieve(
            query="test query",
            k=5,
            filters={"year_min": 2021},
            retriever_instance=custom_engine,
        )

        self.assertIsInstance(res, RetrievalResult)
        self.assertEqual(len(res.chunks), 1)
        self.assertIn("dense_latency_ms", res.latencies_ms)
        self.assertIn("sparse_latency_ms", res.latencies_ms)
        self.assertIn("fusion_latency_ms", res.latencies_ms)
        self.assertIn("hydration_latency_ms", res.latencies_ms)
        self.assertIn("total_latency_ms", res.latencies_ms)


if __name__ == "__main__":
    unittest.main()
