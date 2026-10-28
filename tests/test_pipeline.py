import unittest
from schemas.models import Paper, Chunk, Summary, ComparisonTable, ComparisonRow, GapReport, Hypothesis
from ml_interfaces.protocols import (
    RankerProtocol,
    NoveltyProtocol,
    TrendProtocol,
    ImpactProtocol,
)
from ml_interfaces.stubs import (
    StubRanker,
    StubNoveltyScorer,
    StubTrendForecaster,
    StubImpactPredictor,
)
from ingestion.cleaner import TextCleaner
from ingestion.chunker import Chunker
from ingestion.deduplication import Deduplicator
from retrieval.vector_store import InMemoryVectorStore
from retrieval.retriever import HybridRetriever


class TestPipeline(unittest.TestCase):

    def test_pydantic_schemas_instantiation(self):
        paper = Paper(
            id="arxiv:2305.18290",
            title="Direct Preference Optimization",
            abstract="Direct preference optimization algorithm.",
            authors=["Rafael Rafailov"],
            year=2023,
        )
        self.assertEqual(paper.title, "Direct Preference Optimization")
        self.assertEqual(paper.citation_count, 0)

        chunker = Chunker()
        chunks = chunker.chunk_paper(paper)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0].paper_id, paper.id)

        summary = Summary(
            paper_id=paper.id,
            tl_dr="Presents DPO.",
            key_contributions=["Simpler alignment without reward models"],
            methodology="Closed-form implicit reward derivation",
            limitations=["Subject to reference model drift"],
        )
        self.assertEqual(len(summary.key_contributions), 1)

        gap = GapReport(
            topic="RLHF and Alignment",
            analyzed_paper_ids=[paper.id],
            unaddressed_problems=["Out-of-distribution robustness"],
            synthesis="Significant efficiency gains but requires robustness validation.",
        )
        self.assertEqual(len(gap.analyzed_paper_ids), 1)

        hypo = Hypothesis(
            title="Conservative DPO with Trust Region Regularization",
            statement="Constraining parameter divergence prevents mode collapse.",
            rationale="Rooted in theoretical bounds of KL divergence.",
            grounded_paper_ids=[paper.id],
            target_gap="Reference model drift",
            proposed_methodology="Add adaptive penalty term to DPO loss.",
            testability_and_risks="Verifiable with Anthropic HH-RLHF benchmark.",
        )
        self.assertEqual(hypo.confidence_score, 0.5)

    def test_ml_interfaces_protocols_and_stubs(self):
        ranker: RankerProtocol = StubRanker()
        novelty: NoveltyProtocol = StubNoveltyScorer()
        trend: TrendProtocol = StubTrendForecaster()
        impact: ImpactProtocol = StubImpactPredictor()

        paper = Paper(
            id="test-1",
            title="Attention is All You Need",
            abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
            authors=["Ashish Vaswani", "Noam Shazeer"],
            citation_count=10000,
        )

        scores = ranker.rank("transformer architecture", [paper])
        self.assertEqual(len(scores), 1)
        self.assertTrue(0.0 <= scores[0] <= 1.0)

        nov_val = novelty.score(paper)
        self.assertTrue(0.0 <= nov_val <= 1.0)

        trend_series = trend.forecast("Transformers")
        self.assertEqual(trend_series.topic, "Transformers")
        self.assertEqual(len(trend_series.historical_points), 6)
        self.assertEqual(len(trend_series.forecast_points), 6)

        impact_val = impact.predict(paper)
        self.assertGreaterEqual(impact_val, paper.citation_count)

    def test_retriever_pipeline_with_stubs(self):
        store = InMemoryVectorStore()
        ranker = StubRanker()
        novelty = StubNoveltyScorer()
        impact = StubImpactPredictor()

        p1 = Paper(id="p1", title="SPECTER: Document-level Representation Learning", abstract="Pretrained language models for scientific papers.")
        p2 = Paper(id="p2", title="SciBERT: A Pretrained Language Model for Scientific Text", abstract="BERT trained on computer science and biomedical papers.")

        store.upsert_paper(p1)
        store.upsert_paper(p2)

        retriever = HybridRetriever(
            vector_store=store,
            ranker=ranker,
            novelty_scorer=novelty,
            impact_predictor=impact,
        )

        results = retriever.retrieve("document representations", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertGreaterEqual(results[0].relevance_score, results[1].relevance_score)
        self.assertTrue(hasattr(results[0], "novelty_score"))
        self.assertTrue(hasattr(results[0], "predicted_impact"))


if __name__ == "__main__":
    unittest.main()
