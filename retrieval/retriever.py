from __future__ import annotations

from typing import List, Sequence, Optional
from pydantic import BaseModel, Field

from schemas.models import Paper
from ml_interfaces.protocols import RankerProtocol, NoveltyProtocol, ImpactProtocol
from retrieval.vector_store import InMemoryVectorStore


class EnrichedPaperResult(BaseModel):
    """Paper enriched with ML signal predictions."""
    paper: Paper
    relevance_score: float = Field(..., description="Cross-encoder or dense relevance score.")
    novelty_score: float = Field(..., description="Novelty distance score [0, 1].")
    predicted_impact: float = Field(..., description="Predicted citation count.")


class HybridRetriever:
    """
    RAG retriever implementing the Aurelium pipeline:
    1. Candidate generation via vector/keyword similarity (pgvector).
    2. Relevance reranking via RankerProtocol (cross-encoder or ML ranker).
    3. Multi-signal enrichment via NoveltyProtocol and ImpactProtocol.
    """

    def __init__(
        self,
        vector_store: InMemoryVectorStore,
        ranker: RankerProtocol,
        novelty_scorer: NoveltyProtocol,
        impact_predictor: ImpactProtocol,
    ):
        self.vector_store = vector_store
        self.ranker = ranker
        self.novelty_scorer = novelty_scorer
        self.impact_predictor = impact_predictor

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        candidate_pool_size: int = 30,
    ) -> List[EnrichedPaperResult]:
        # Step 1: Candidate generation from vector store
        candidates_with_scores = self.vector_store.search_candidates(
            query=query,
            top_k=candidate_pool_size,
        )
        if not candidates_with_scores:
            return []

        candidate_papers = [p for p, _ in candidates_with_scores]

        # Step 2: Relevance reranking via RankerProtocol
        relevance_scores = self.ranker.rank(query=query, papers=candidate_papers)

        # Step 3: Attach ML insights (Novelty + Predicted Impact)
        enriched: List[EnrichedPaperResult] = []
        for paper, rel_score in zip(candidate_papers, relevance_scores):
            novelty = self.novelty_scorer.score(paper)
            impact = self.impact_predictor.predict(paper)
            enriched.append(
                EnrichedPaperResult(
                    paper=paper,
                    relevance_score=rel_score,
                    novelty_score=novelty,
                    predicted_impact=impact,
                )
            )

        # Sort descending by relevance score
        enriched.sort(key=lambda x: x.relevance_score, reverse=True)
        return enriched[:top_k]
