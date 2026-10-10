from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable
from contracts.schemas.models import Paper, TrendSeries


@runtime_checkable
class RankerProtocol(Protocol):
    """
    Protocol for relevance ranking / reranking models.
    Provided by ml/ and consumed by ai/ retrieval.
    """

    def rank(self, query: str, papers: Sequence[Paper]) -> list[float]:
        """
        Rank candidate papers against a search query.

        Args:
            query: The user research query or question.
            papers: Candidate papers retrieved during candidate generation.

        Returns:
            list[float]: Relevance scores in [0.0, 1.0], ordered parallel to `papers`.
        """
        ...


@runtime_checkable
class NoveltyProtocol(Protocol):
    """
    Protocol for paper novelty scoring.
    Provided by ml/ and consumed by ai/ agents.
    Computed as 1 - mean cosine similarity to prior k-NN to avoid future-leakage.
    """

    def score(self, paper: Paper) -> float:
        """
        Score the novelty of a paper relative to historical literature in vector space.

        Args:
            paper: Paper instance with abstract/title and optional precomputed embedding.

        Returns:
            float: Novelty score in [0.0, 1.0]. Higher values indicate greater divergence from prior work.
        """
        ...


@runtime_checkable
class ImpactProtocol(Protocol):
    """
    Protocol for citation impact prediction.
    Provided by ml/ and consumed by ai/ agents.
    """

    def predict(self, paper: Paper) -> float:
        """
        Predict future citation impact / volume for a paper.

        Args:
            paper: Paper instance with metadata (authors, venue, citation_count, embedding).

        Returns:
            float: Predicted citation count or impact metric.
        """
        ...


@runtime_checkable
class TrendProtocol(Protocol):
    """
    Protocol for research topic trend forecasting.
    Provided by ml/ and consumed by ai/ agents.
    """

    def forecast(self, topic: str) -> TrendSeries:
        """
        Forecast trajectory for a given research topic or semantic cluster.

        Args:
            topic: Topic string or cluster identifier.

        Returns:
            TrendSeries: Trajectory with historical points, future forecast points, and emerging indicator.
        """
        ...
