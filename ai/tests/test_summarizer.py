import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock
from contracts.schemas.models import Chunk, Summary as ContractSummary
from ai.agents.cache import LLMCache
from ai.agents.models import (
    BatchVerificationResponse,
    ClaimCitation,
    ClaimVerificationResult,
    MultiPaperSummaryResult,
    PaperSummary,
)
from ai.agents.summarizer import SummarizationAgent


class TestSummarizerAgent(unittest.IsolatedAsyncioTestCase):
    """
    Test suite for the SummarizationAgent.
    Validates:
    - Structured JSON schema validation across 5 scientific dimensions
    - Verifier pass removing unsupported claims and marking partially supported ones
    - Preservation and fallback handling for 'not reported' statements
    - Deterministic LLM response caching
    - Concurrency-controlled multi-paper summarization and cross-paper overview synthesis
    - Conversion to contracts.schemas.models.Summary
    """

    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_client.model = "grok-beta"
        self.cache = LLMCache(enabled=True)
        self.agent = SummarizationAgent(
            llm_client=self.mock_client,
            cache=self.cache,
            concurrency_limit=3,
        )

        self.sample_chunks = [
            Chunk(
                chunk_id="chunk-001",
                paper_id="arxiv:2305.18290",
                section="Introduction",
                text="Direct Preference Optimization (DPO) optimizes language models directly from human preferences without training a reward model.",
                char_start=0,
                char_end=130,
            ),
            Chunk(
                chunk_id="chunk-002",
                paper_id="arxiv:2305.18290",
                section="Experiments",
                text="We evaluate DPO on TL;DR summarization and Anthropic HH benchmarks, demonstrating improved stability over PPO.",
                char_start=131,
                char_end=260,
            ),
        ]

    async def test_schema_validation_and_grounded_claims(self):
        """Tests that initial structured summary parses correctly with grounded claims."""
        raw_summary = PaperSummary(
            paper_id="arxiv:2305.18290",
            problem=[
                ClaimCitation(
                    statement="RLHF with PPO is unstable and computationally complex.",
                    chunk_ids=["chunk-001"],
                )
            ],
            method=[
                ClaimCitation(
                    statement="DPO derives a closed-form policy directly matching preference data.",
                    chunk_ids=["chunk-001"],
                )
            ],
            datasets=[
                ClaimCitation(
                    statement="Evaluated on Anthropic HH and TL;DR summarization.",
                    chunk_ids=["chunk-002"],
                )
            ],
            results=[
                ClaimCitation(
                    statement="DPO outperforms PPO on win-rate benchmarks.",
                    chunk_ids=["chunk-002"],
                )
            ],
            limitations=[
                ClaimCitation(
                    statement="Susceptible to out-of-distribution reference drift.",
                    chunk_ids=["chunk-002"],
                )
            ],
        )

        # Mock summarizer call
        self.mock_client.complete_json = AsyncMock()

        # Batch verifier verdicts: all supported
        verifier_response = BatchVerificationResponse(
            verdicts=[
                ClaimVerificationResult(
                    claim_index=0,
                    dimension="problem",
                    statement="RLHF with PPO is unstable and computationally complex.",
                    status="supported",
                    reason="Supported by chunk-001.",
                ),
                ClaimVerificationResult(
                    claim_index=1,
                    dimension="method",
                    statement="DPO derives a closed-form policy directly matching preference data.",
                    status="supported",
                    reason="Supported by chunk-001.",
                ),
                ClaimVerificationResult(
                    claim_index=2,
                    dimension="datasets",
                    statement="Evaluated on Anthropic HH and TL;DR summarization.",
                    status="supported",
                    reason="Directly cited in chunk-002.",
                ),
                ClaimVerificationResult(
                    claim_index=3,
                    dimension="results",
                    statement="DPO outperforms PPO on win-rate benchmarks.",
                    status="supported",
                    reason="Empirical findings match chunk-002.",
                ),
                ClaimVerificationResult(
                    claim_index=4,
                    dimension="limitations",
                    statement="Susceptible to out-of-distribution reference drift.",
                    status="supported",
                    reason="Stated limitation in chunk-002.",
                ),
            ]
        )

        self.mock_client.complete_json.side_effect = [raw_summary, verifier_response]

        res = await self.agent.summarize_paper("arxiv:2305.18290", chunks=self.sample_chunks)

        self.assertEqual(res.paper_id, "arxiv:2305.18290")
        self.assertEqual(len(res.problem), 1)
        self.assertEqual(res.problem[0].chunk_ids, ["chunk-001"])
        self.assertEqual(res.problem[0].verification_status, "supported")
        self.assertEqual(len(res.results), 1)
        self.assertEqual(res.results[0].verification_status, "supported")
        self.assertEqual(len(res.verification_log), 5)

    async def test_verifier_removes_unsupported_and_marks_partially_supported(self):
        """Tests that unsupported claims are removed and partially supported claims are marked."""
        raw_summary = PaperSummary(
            paper_id="arxiv:2305.18290",
            problem=[
                ClaimCitation(
                    statement="RLHF with PPO is unstable.",
                    chunk_ids=["chunk-001"],
                )
            ],
            method=[
                ClaimCitation(
                    statement="DPO matches preference data.",
                    chunk_ids=["chunk-001"],
                )
            ],
            datasets=[
                ClaimCitation(
                    statement="Evaluated on TL;DR.",
                    chunk_ids=["chunk-002"],
                )
            ],
            results=[
                ClaimCitation(
                    statement="DPO achieves 99% accuracy on ImageNet.",  # Hallucinated!
                    chunk_ids=["chunk-002"],
                ),
                ClaimCitation(
                    statement="DPO outperforms PPO with modest compute savings.",
                    chunk_ids=["chunk-002"],
                ),
            ],
            limitations=[
                ClaimCitation(statement="not reported", chunk_ids=[]),
            ],
        )

        verifier_response = BatchVerificationResponse(
            verdicts=[
                ClaimVerificationResult(
                    claim_index=0,
                    dimension="problem",
                    statement="RLHF with PPO is unstable.",
                    status="supported",
                    reason="Supported.",
                ),
                ClaimVerificationResult(
                    claim_index=1,
                    dimension="method",
                    statement="DPO matches preference data.",
                    status="supported",
                    reason="Supported.",
                ),
                ClaimVerificationResult(
                    claim_index=2,
                    dimension="datasets",
                    statement="Evaluated on TL;DR.",
                    status="supported",
                    reason="Supported.",
                ),
                ClaimVerificationResult(
                    claim_index=3,
                    dimension="results",
                    statement="DPO achieves 99% accuracy on ImageNet.",
                    status="unsupported",
                    reason="ImageNet was never evaluated in this paper.",
                ),
                ClaimVerificationResult(
                    claim_index=4,
                    dimension="results",
                    statement="DPO outperforms PPO with modest compute savings.",
                    status="partially_supported",
                    reason="Paper notes stability and elimination of reward modeling, but exact compute percentage is unspecified.",
                ),
            ]
        )

        self.mock_client.complete_json = AsyncMock(side_effect=[raw_summary, verifier_response])

        res = await self.agent.summarize_paper("arxiv:2305.18290", chunks=self.sample_chunks)

        # Unsupported ImageNet claim must be removed from results
        self.assertEqual(len(res.results), 1)
        self.assertEqual(res.results[0].statement, "DPO outperforms PPO with modest compute savings.")
        self.assertEqual(res.results[0].verification_status, "partially_supported")
        self.assertIn("compute percentage", res.results[0].verification_reason)

        # Verification log should record both evaluated result claims
        unsupported_logs = [log for log in res.verification_log if log["status"] == "unsupported"]
        self.assertEqual(len(unsupported_logs), 1)
        self.assertIn("ImageNet", unsupported_logs[0]["statement"])

    async def test_not_reported_handling_and_fallback(self):
        """
        Tests that:
        1. Explicit 'not reported' statements are preserved with empty chunk_ids and not sent to verifier.
        2. Dimensions where all claims were unsupported drop to 'not reported'.
        """
        raw_summary = PaperSummary(
            paper_id="arxiv:2305.18290",
            problem=[
                ClaimCitation(
                    statement="Hallucinated problem statement.",
                    chunk_ids=["chunk-001"],
                )
            ],
            method=[ClaimCitation(statement="not reported", chunk_ids=[])],
            datasets=[ClaimCitation(statement="not reported", chunk_ids=[])],
            results=[ClaimCitation(statement="not reported", chunk_ids=[])],
            limitations=[ClaimCitation(statement="not reported", chunk_ids=[])],
        )

        verifier_response = BatchVerificationResponse(
            verdicts=[
                ClaimVerificationResult(
                    claim_index=0,
                    dimension="problem",
                    statement="Hallucinated problem statement.",
                    status="unsupported",
                    reason="Not in text.",
                ),
            ]
        )

        self.mock_client.complete_json = AsyncMock(side_effect=[raw_summary, verifier_response])

        res = await self.agent.summarize_paper("arxiv:2305.18290", chunks=self.sample_chunks)

        # Problem had only an unsupported claim; after removal, it must fall back to 'not reported'
        self.assertEqual(len(res.problem), 1)
        self.assertEqual(res.problem[0].statement, "not reported")
        self.assertEqual(res.problem[0].chunk_ids, [])

        # Dimensions that started as 'not reported' must remain untouched
        for dim in [res.method, res.datasets, res.results, res.limitations]:
            self.assertEqual(len(dim), 1)
            self.assertEqual(dim[0].statement, "not reported")
            self.assertEqual(dim[0].chunk_ids, [])

    async def test_nonexistent_chunk_id_immediately_unsupported(self):
        """Claims citing non-existent chunk IDs are immediately dropped without extra LLM call."""
        raw_summary = PaperSummary(
            paper_id="arxiv:2305.18290",
            problem=[
                ClaimCitation(
                    statement="Statement citing phantom chunk.",
                    chunk_ids=["non-existent-chunk-999"],
                )
            ],
            method=[ClaimCitation(statement="not reported", chunk_ids=[])],
            datasets=[ClaimCitation(statement="not reported", chunk_ids=[])],
            results=[ClaimCitation(statement="not reported", chunk_ids=[])],
            limitations=[ClaimCitation(statement="not reported", chunk_ids=[])],
        )

        # Verifier LLM should NOT even be called because all candidate citations are invalid
        self.mock_client.complete_json = AsyncMock(return_value=raw_summary)

        res = await self.agent.summarize_paper("arxiv:2305.18290", chunks=self.sample_chunks)

        self.assertEqual(res.problem[0].statement, "not reported")
        self.assertEqual(self.mock_client.complete_json.await_count, 1)  # only initial summarizer
        self.assertEqual(res.verification_log[0]["status"], "unsupported")
        self.assertIn("do not exist", res.verification_log[0]["reason"])

    async def test_deterministic_llm_caching(self):
        """Tests that identical inputs hit the LLM cache and avoid duplicate API calls."""
        raw_summary = PaperSummary(
            paper_id="arxiv:2305.18290",
            problem=[
                ClaimCitation(
                    statement="RLHF instability problem.",
                    chunk_ids=["chunk-001"],
                )
            ],
            method=[ClaimCitation(statement="not reported", chunk_ids=[])],
            datasets=[ClaimCitation(statement="not reported", chunk_ids=[])],
            results=[ClaimCitation(statement="not reported", chunk_ids=[])],
            limitations=[ClaimCitation(statement="not reported", chunk_ids=[])],
        )
        verifier_response = BatchVerificationResponse(
            verdicts=[
                ClaimVerificationResult(
                    claim_index=0,
                    dimension="problem",
                    statement="RLHF instability problem.",
                    status="supported",
                    reason="Grounded in chunk-001.",
                )
            ]
        )

        self.mock_client.complete_json = AsyncMock(side_effect=[raw_summary, verifier_response])

        # First call: populates cache
        res1 = await self.agent.summarize_paper("arxiv:2305.18290", chunks=self.sample_chunks)
        self.assertEqual(self.mock_client.complete_json.await_count, 2)
        self.assertGreater(self.cache.size(), 0)

        # Second call with identical input: should hit cache
        res2 = await self.agent.summarize_paper("arxiv:2305.18290", chunks=self.sample_chunks)
        # Call count must NOT increase
        self.assertEqual(self.mock_client.complete_json.await_count, 2)
        self.assertEqual(res1.problem[0].statement, res2.problem[0].statement)

    async def test_batch_summarization_and_overview(self):
        """Tests multi-paper concurrent summarization and bracketed citation in cross-paper synthesis."""
        paper1_chunks = [
            Chunk(
                chunk_id="p1-c1",
                paper_id="paper-1",
                section="Intro",
                text="Paper 1 introduces technique A.",
                char_start=0,
                char_end=50,
            )
        ]
        paper2_chunks = [
            Chunk(
                chunk_id="p2-c1",
                paper_id="paper-2",
                section="Intro",
                text="Paper 2 introduces technique B.",
                char_start=0,
                char_end=50,
            )
        ]

        summary1 = PaperSummary(
            paper_id="paper-1",
            problem=[ClaimCitation(statement="Problem 1", chunk_ids=["p1-c1"])],
            method=[ClaimCitation(statement="Technique A", chunk_ids=["p1-c1"])],
            datasets=[ClaimCitation(statement="not reported", chunk_ids=[])],
            results=[ClaimCitation(statement="not reported", chunk_ids=[])],
            limitations=[ClaimCitation(statement="not reported", chunk_ids=[])],
        )
        summary2 = PaperSummary(
            paper_id="paper-2",
            problem=[ClaimCitation(statement="Problem 2", chunk_ids=["p2-c1"])],
            method=[ClaimCitation(statement="Technique B", chunk_ids=["p2-c1"])],
            datasets=[ClaimCitation(statement="not reported", chunk_ids=[])],
            results=[ClaimCitation(statement="not reported", chunk_ids=[])],
            limitations=[ClaimCitation(statement="not reported", chunk_ids=[])],
        )

        verifier_resp1 = BatchVerificationResponse(
            verdicts=[
                ClaimVerificationResult(
                    claim_index=0,
                    dimension="problem",
                    statement="Problem 1",
                    status="supported",
                    reason="OK",
                ),
                ClaimVerificationResult(
                    claim_index=1,
                    dimension="method",
                    statement="Technique A",
                    status="supported",
                    reason="OK",
                ),
            ]
        )
        verifier_resp2 = BatchVerificationResponse(
            verdicts=[
                ClaimVerificationResult(
                    claim_index=0,
                    dimension="problem",
                    statement="Problem 2",
                    status="supported",
                    reason="OK",
                ),
                ClaimVerificationResult(
                    claim_index=1,
                    dimension="method",
                    statement="Technique B",
                    status="supported",
                    reason="OK",
                ),
            ]
        )

        # complete_json will be called 4 times (2 summarizer + 2 verifier)
        self.mock_client.complete_json = AsyncMock(
            side_effect=[summary1, verifier_resp1, summary2, verifier_resp2]
        )
        # complete will be called 1 time for cross-paper overview
        self.mock_client.complete = AsyncMock(
            return_value="Paper [paper-1] investigates Problem 1 via Technique A, while [paper-2] addresses Problem 2 via Technique B."
        )

        batch_result = await self.agent.summarize_papers(
            paper_ids=["paper-1", "paper-2"],
            chunks_by_paper={"paper-1": paper1_chunks, "paper-2": paper2_chunks},
            max_concurrency=2,
        )

        self.assertIsInstance(batch_result, MultiPaperSummaryResult)
        self.assertEqual(len(batch_result.summaries), 2)
        self.assertIn("[paper-1]", batch_result.cross_paper_overview)
        self.assertIn("[paper-2]", batch_result.cross_paper_overview)
        self.mock_client.complete.assert_awaited_once()

    def test_to_contract_summary_conversion(self):
        """Tests that PaperSummary safely converts to contracts.schemas.models.Summary."""
        summary = PaperSummary(
            paper_id="arxiv:2305.18290",
            problem=[
                ClaimCitation(statement="RLHF instability.", chunk_ids=["c1", "c2"])
            ],
            method=[
                ClaimCitation(statement="Direct Preference Optimization.", chunk_ids=["c2", "c3"])
            ],
            datasets=[
                ClaimCitation(statement="TL;DR corpus.", chunk_ids=["c4"])
            ],
            results=[
                ClaimCitation(statement="Outperforms PPO on win rate.", chunk_ids=["c4"])
            ],
            limitations=[
                ClaimCitation(statement="Reference policy sensitivity.", chunk_ids=["c5"])
            ],
        )

        contract_summary = summary.to_contract_summary()
        self.assertIsInstance(contract_summary, ContractSummary)
        self.assertEqual(contract_summary.paper_id, "arxiv:2305.18290")
        self.assertIn("RLHF instability", contract_summary.tl_dr)
        self.assertIn("Outperforms PPO", contract_summary.tl_dr)
        self.assertIn("Direct Preference Optimization.", contract_summary.key_contributions)
        self.assertIn("Reference policy sensitivity.", contract_summary.limitations)
        self.assertEqual(contract_summary.citations, ["c1", "c2", "c3", "c4", "c5"])


if __name__ == "__main__":
    unittest.main()
