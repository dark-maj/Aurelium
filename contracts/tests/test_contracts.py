from datetime import datetime, timezone
import unittest

from contracts.schemas.models import (
    Paper,
    Chunk,
    Summary,
    Claim,
    ComparisonCell,
    ComparisonTable,
    GapReport,
    Hypothesis,
    NoveltyScore,
    TrendPoint,
    TrendSeries,
    ResearchResult,
)
from contracts.interfaces.protocols import (
    RankerProtocol,
    NoveltyProtocol,
    ImpactProtocol,
    TrendProtocol,
)
from contracts.interfaces.stubs import (
    StubRanker,
    StubNoveltyScorer,
    StubImpactPredictor,
    StubTrendForecaster,
)


class TestProtocolsAndStubs(unittest.TestCase):
    """Verifies that stub implementations strictly satisfy their respective typing.Protocols."""

    def setUp(self):
        self.sample_paper = Paper(
            id="arxiv:2305.18290",
            title="Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
            abstract="We show how to optimize language models on human preferences directly without training an explicit reward model.",
            authors=["Rafael Rafailov", "Archit Sharma", "Eric Mitchell", "Stefano Ermon", "Christopher D. Manning", "Chelsea Finn"],
            venue="NeurIPS 2023",
            year=2023,
            published_date=datetime(2023, 5, 29, 0, 0, 0, tzinfo=timezone.utc),
            doi="10.48550/arXiv.2305.18290",
            arxiv_id="2305.18290",
            semantic_scholar_id="204e3073870fae3f05bcbc65834984aa3125cc64",
            citation_count=1200,
            fields_of_study=["Computer Science", "Machine Learning"],
        )

    def test_stub_ranker_protocol_compliance(self):
        ranker = StubRanker()
        self.assertTrue(isinstance(ranker, RankerProtocol), "StubRanker must implement RankerProtocol")

        scores = ranker.rank("direct preference optimization", [self.sample_paper])
        self.assertIsInstance(scores, list)
        self.assertEqual(len(scores), 1)
        self.assertIsInstance(scores[0], float)
        self.assertTrue(0.0 <= scores[0] <= 1.0)

        # Verify determinism
        scores_repeat = ranker.rank("direct preference optimization", [self.sample_paper])
        self.assertEqual(scores, scores_repeat)

    def test_stub_novelty_protocol_compliance(self):
        scorer = StubNoveltyScorer()
        self.assertTrue(isinstance(scorer, NoveltyProtocol), "StubNoveltyScorer must implement NoveltyProtocol")

        score = scorer.score(self.sample_paper)
        self.assertIsInstance(score, float)
        self.assertTrue(0.0 <= score <= 1.0)

        # Verify determinism
        score_repeat = scorer.score(self.sample_paper)
        self.assertEqual(score, score_repeat)

    def test_stub_impact_protocol_compliance(self):
        predictor = StubImpactPredictor()
        self.assertTrue(isinstance(predictor, ImpactProtocol), "StubImpactPredictor must implement ImpactProtocol")

        predicted_citations = predictor.predict(self.sample_paper)
        self.assertIsInstance(predicted_citations, float)
        self.assertGreaterEqual(predicted_citations, 0.0)

        # Verify determinism
        repeat = predictor.predict(self.sample_paper)
        self.assertEqual(predicted_citations, repeat)

    def test_stub_trend_protocol_compliance(self):
        forecaster = StubTrendForecaster()
        self.assertTrue(isinstance(forecaster, TrendProtocol), "StubTrendForecaster must implement TrendProtocol")

        series = forecaster.forecast("Preference Optimization")
        self.assertIsInstance(series, TrendSeries)
        self.assertEqual(series.topic, "Preference Optimization")
        self.assertEqual(len(series.historical_points), 6)
        self.assertEqual(len(series.forecast_points), 6)
        self.assertIsInstance(series.growth_rate, float)
        self.assertIsInstance(series.is_emerging, bool)

        # Verify determinism
        series_repeat = forecaster.forecast("Preference Optimization")
        self.assertEqual(series, series_repeat)


class TestSchemasJsonRoundtrip(unittest.TestCase):
    """Verifies that all 10 Pydantic v2 domain schemas serialize to and deserialize from JSON cleanly."""

    def test_paper_schema_roundtrip(self):
        paper = Paper(
            id="arxiv:1706.03762",
            title="Attention Is All You Need",
            abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
            authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"],
            venue="NeurIPS 2017",
            year=2017,
            published_date=datetime(2017, 6, 12, 0, 0, 0, tzinfo=timezone.utc),
            doi="10.48550/arXiv.1706.03762",
            arxiv_id="1706.03762",
            semantic_scholar_id="204e3073870fae3f05bcbc65834984aa3125cc64",
            citation_count=95000,
            fields_of_study=["Computer Science", "Artificial Intelligence"],
            url="https://arxiv.org/abs/1706.03762",
            embedding=[0.012, -0.045, 0.982],
            metadata={"source": "arxiv", "license": "cc-by"},
        )
        json_data = paper.model_dump_json()
        recovered = Paper.model_validate_json(json_data)
        self.assertEqual(paper, recovered)
        self.assertEqual(recovered.doi, "10.48550/arXiv.1706.03762")
        self.assertEqual(recovered.arxiv_id, "1706.03762")
        self.assertEqual(recovered.semantic_scholar_id, "204e3073870fae3f05bcbc65834984aa3125cc64")
        self.assertEqual(recovered.published_date, datetime(2017, 6, 12, 0, 0, 0, tzinfo=timezone.utc))

    def test_chunk_schema_roundtrip(self):
        chunk = Chunk(
            chunk_id="chunk_1706_001",
            paper_id="arxiv:1706.03762",
            section="Abstract",
            text="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
            char_start=0,
            char_end=112,
            token_count=16,
            embedding=[0.1, -0.2, 0.3],
            metadata={"page": 1},
        )
        json_data = chunk.model_dump_json()
        recovered = Chunk.model_validate_json(json_data)
        self.assertEqual(chunk, recovered)
        self.assertEqual(recovered.chunk_id, "chunk_1706_001")
        self.assertEqual(recovered.paper_id, "arxiv:1706.03762")
        self.assertEqual(recovered.section, "Abstract")
        self.assertEqual(recovered.char_start, 0)
        self.assertEqual(recovered.char_end, 112)

    def test_summary_schema_roundtrip(self):
        summary = Summary(
            paper_id="arxiv:2305.18290",
            tl_dr="Proposes direct preference optimization without separate reward modeling.",
            key_contributions=["Closed-form derivation of optimal policy under Bradley-Terry preferences", "Eliminates RL instability in LLM alignment"],
            methodology="Implicit reward parameterization mapped directly to token cross-entropy loss.",
            limitations=["Requires paired preference dataset", "Susceptible to reference model collapse if unconstrained"],
            citations=["chunk_2305_001", "chunk_2305_004"],
            novelty_assessment="High conceptual departure from standard RLHF PPO pipelines.",
            predicted_impact_assessment="Expected high adoption in foundation model alignment.",
        )
        json_data = summary.model_dump_json()
        recovered = Summary.model_validate_json(json_data)
        self.assertEqual(summary, recovered)

    def test_claim_schema_roundtrip(self):
        claim = Claim(
            id="claim_001",
            statement="DPO eliminates the need for fitting a standalone reward model.",
            paper_id="arxiv:2305.18290",
            chunk_id="chunk_2305_001",
            quote="We show how to optimize language models directly from preferences without explicit reward modeling.",
            confidence=0.98,
        )
        json_data = claim.model_dump_json()
        recovered = Claim.model_validate_json(json_data)
        self.assertEqual(claim, recovered)
        self.assertEqual(recovered.confidence, 0.98)
        self.assertEqual(recovered.chunk_id, "chunk_2305_001")

    def test_comparison_cell_and_table_roundtrip(self):
        cell_1 = ComparisonCell(
            paper_id="paper_a",
            dimension="Methodology",
            value="PPO with explicit reward model",
            citation_chunk_ids=["chunk_a_1"],
        )
        cell_2 = ComparisonCell(
            paper_id="paper_b",
            dimension="Methodology",
            value="Direct token likelihood optimization",
            citation_chunk_ids=["chunk_b_1"],
        )
        table = ComparisonTable(
            topic="LLM Preference Alignment",
            dimensions=["Methodology"],
            paper_ids=["paper_a", "paper_b"],
            cells=[cell_1, cell_2],
            synthesis="Paper B removes reinforcement learning overhead at the expense of requiring offline preference pairs.",
        )
        json_data = table.model_dump_json()
        recovered = ComparisonTable.model_validate_json(json_data)
        self.assertEqual(table, recovered)
        self.assertEqual(len(recovered.cells), 2)
        self.assertEqual(recovered.cells[0].dimension, "Methodology")

    def test_gap_report_roundtrip(self):
        report = GapReport(
            topic="RLHF Robustness",
            analyzed_paper_ids=["paper_a", "paper_b"],
            unaddressed_problems=["Out-of-distribution adversarial prompts degrade preference alignment."],
            contradictions_or_conflicts=["Paper A claims KL penalty prevents reward hacking, while Paper B observes drift."],
            under_explored_avenues=["Online iterative preference synthesis using multi-agent debate."],
            methodological_bottlenecks=["Dependence on static, potentially contaminated preference benchmarks."],
            synthesis="Significant empirical progress has been made, but safety against adversarial distribution shifts remains an open gap.",
            citation_chunk_ids=["chunk_a_3", "chunk_b_7"],
        )
        json_data = report.model_dump_json()
        recovered = GapReport.model_validate_json(json_data)
        self.assertEqual(report, recovered)

    def test_hypothesis_roundtrip(self):
        hypo = Hypothesis(
            id="hypo_001",
            title="Curvature-regularized direct alignment",
            statement="Adding Riemannian curvature regularization to the DPO objective reduces mode collapse under distribution shift by 25%.",
            rationale="Addresses the reference model drift identified in the gap report.",
            grounded_paper_ids=["paper_b"],
            grounded_chunk_ids=["chunk_b_12"],
            target_gap="Reference model drift under out-of-distribution prompts",
            proposed_methodology="Evaluate on synthetic adversarial jailbreak benchmarks with varying curvature penalties.",
            expected_outcomes=["Lower variance in token likelihoods", "Increased robustness on held-out safety benchmarks"],
            testability_and_risks="Directly verifiable on standard LLM benchmarks; risk of slight degradation in creative benchmarks.",
            confidence_score=0.85,
        )
        json_data = hypo.model_dump_json()
        recovered = Hypothesis.model_validate_json(json_data)
        self.assertEqual(hypo, recovered)
        self.assertEqual(recovered.confidence_score, 0.85)

    def test_novelty_score_roundtrip(self):
        novelty = NoveltyScore(
            paper_id="paper_x",
            score=0.8245,
            nearest_prior_distance=0.8245,
            percentile=94.5,
            prior_papers_count=520,
        )
        json_data = novelty.model_dump_json()
        recovered = NoveltyScore.model_validate_json(json_data)
        self.assertEqual(novelty, recovered)

    def test_trend_series_roundtrip(self):
        series = TrendSeries(
            topic="Direct Preference Optimization",
            historical_points=[
                TrendPoint(timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc), value=12.0, lower_bound=10.0, upper_bound=14.0),
                TrendPoint(timestamp=datetime(2025, 2, 1, tzinfo=timezone.utc), value=18.5, lower_bound=16.0, upper_bound=21.0),
            ],
            forecast_points=[
                TrendPoint(timestamp=datetime(2025, 3, 1, tzinfo=timezone.utc), value=26.0, lower_bound=22.0, upper_bound=30.0),
            ],
            growth_rate=45.2,
            is_emerging=True,
        )
        json_data = series.model_dump_json()
        recovered = TrendSeries.model_validate_json(json_data)
        self.assertEqual(series, recovered)

    def test_research_result_roundtrip(self):
        paper = Paper(
            id="p1",
            title="DPO Paper",
            abstract="Abstract of DPO",
            authors=["Author 1"],
            year=2023,
            citation_count=50,
        )
        result = ResearchResult(
            query="preference optimization",
            ranked_papers=[paper],
            ranking_scores=[0.95],
            summaries=[
                Summary(
                    paper_id="p1",
                    tl_dr="TLDR of DPO",
                    key_contributions=["Contr 1"],
                    methodology="Method 1",
                    limitations=["Limit 1"],
                )
            ],
            novelty_scores=[NoveltyScore(paper_id="p1", score=0.75)],
            synthesis="Comprehensive synthesis answering preference optimization.",
        )
        json_data = result.model_dump_json()
        recovered = ResearchResult.model_validate_json(json_data)
        self.assertEqual(result, recovered)


if __name__ == "__main__":
    unittest.main()
