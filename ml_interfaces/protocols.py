from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable
from schemas.models import Paper
from ml_interfaces.types import TrendSeries


@runtime_checkable
class RankerProtocol(Protocol):
    """
    Protocol for relevance ranking / reranking models.
    Takes a query and candidate papers, returning relevance scores (cross-encoder or dense cosine).
    """

    def rank(self, query: str, papers: Sequence[Paper]) -> list[float]:
        """
        Rank candidate papers against the user query.

        Args:
            query: The search query, question, or research premise.
            papers: Candidate papers retrieved from candidate generation (e.g. pgvector).

        Returns:
            list[float]: A list of relevance scores parallel to the input papers list.
        """
        ...


@runtime_checkable
class NoveltyProtocol(Protocol):
    """
    Protocol for paper novelty scoring.
    Computes distance to historical neighborhood in embedding space
    (e.g., 1 - mean cosine similarity to k-nearest prior neighbors).
    """

    def score(self, paper: Paper) -> float:
        """
        Score how novel a paper is relative to prior literature in vector space.

        Args:
            paper: Paper instance with abstract/title and optional precomputed embedding.

        Returns:
            float: Novelty score in [0.0, 1.0], where 1.0 represents high divergence from prior work.
        """
        ...


@runtime_checkable
class TrendProtocol(Protocol):
    """
    Protocol for research topic trend forecasting.
    Uses time-series models (Prophet/ARIMA over HDBSCAN cluster aggregations)
    to forecast growth trajectory.
    """

    def forecast(self, topic: str) -> TrendSeries:
        """
        Forecast trajectory for a given research topic or semantic cluster.

        Args:
            topic: Topic string or semantic cluster identifier.

        Returns:
            TrendSeries: Historical and forecasted trend points with growth rate indicator.
        """
        ...


@runtime_checkable
class ImpactProtocol(Protocol):
    """
    Protocol for citation impact prediction.
    Predicts future citations / log-citations using paper embedding + metadata features
    (authors, venue, novelty score).
    """

    def predict(self, paper: Paper) -> float:
        """
        Predict citation impact / expected citation volume.

        Args:
            paper: Paper instance with metadata and embedding.

        Returns:
            float: Predicted citation count or percentile impact metric.
        """
        ...
