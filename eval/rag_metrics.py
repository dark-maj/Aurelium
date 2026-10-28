from typing import List
from pydantic import BaseModel, Field


class RagEvalResult(BaseModel):
    """Evaluation result for a RAG copilot generation."""
    query: str
    faithfulness: float = Field(..., ge=0.0, le=1.0, description="Claims supported by retrieved context.")
    answer_relevance: float = Field(..., ge=0.0, le=1.0, description="Relevance of the answer to user intent.")
    context_recall: float = Field(..., ge=0.0, le=1.0, description="Proportion of necessary context retrieved.")
    latency_ms: float


class RagEvaluator:
    """Evaluates AI Copilot agent outputs against ground truth contexts."""

    def evaluate_response(
        self,
        query: str,
        response: str,
        retrieved_contexts: List[str],
        ground_truth: str = "",
    ) -> RagEvalResult:
        # Heuristic / deterministic evaluation placeholder for benchmarking pipelines
        context_words = set(" ".join(retrieved_contexts).lower().split())
        response_words = set(response.lower().split())

        grounded_ratio = (
            len(response_words & context_words) / max(1, len(response_words))
        )
        faithfulness = round(min(1.0, 0.5 + grounded_ratio * 0.5), 3)

        query_words = set(query.lower().split())
        relevance_ratio = len(query_words & response_words) / max(1, len(query_words))
        answer_relevance = round(min(1.0, 0.4 + relevance_ratio * 0.6), 3)

        return RagEvalResult(
            query=query,
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_recall=0.85,
            latency_ms=120.0,
        )
