from __future__ import annotations

import hashlib
import math
from datetime import datetime, timedelta, timezone
from typing import Sequence

from schemas.models import Paper
from ml_interfaces.protocols import (
    RankerProtocol,
    NoveltyProtocol,
    TrendProtocol,
    ImpactProtocol,
)
from ml_interfaces.types import TrendPoint, TrendSeries


def _deterministic_float(seed: str) -> float:
    """Generate a reproducible float in [0.0, 1.0) given a string seed."""
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


class StubRanker(RankerProtocol):
    """
    Mock ranker returning deterministic synthetic relevance scores.
    Simulates cross-encoder scoring based on query-title term overlap and hash dispersion.
    """

    def rank(self, query: str, papers: Sequence[Paper]) -> list[float]:
        query_words = set(query.lower().split())
        scores: list[float] = []

        for paper in papers:
            title_words = set(paper.title.lower().split())
            abstract_words = set(paper.abstract.lower().split())
            overlap = len(query_words & (title_words | abstract_words))

            # Base score from keyword overlap + pseudo-random semantic similarity
            base_score = min(0.6, overlap * 0.15)
            pseudo_semantic = _deterministic_float(f"{query}:{paper.id}") * 0.4
            final_score = round(min(1.0, max(0.05, base_score + pseudo_semantic)), 4)
            scores.append(final_score)

        return scores


class StubNoveltyScorer(NoveltyProtocol):
    """
    Mock novelty scorer returning synthetic distance in [0.1, 0.95].
    Simulates embedding neighborhood distance (1 - mean cosine similarity to prior papers).
    """

    def score(self, paper: Paper) -> float:
        # Generate predictable novelty based on paper attributes
        pseudo_dist = _deterministic_float(f"novelty:{paper.id}:{paper.title}")
        # Scale to typical range: [0.15, 0.92]
        return round(0.15 + (pseudo_dist * 0.77), 4)


class StubTrendForecaster(TrendProtocol):
    """
    Mock trend forecaster returning synthetic time series with monthly buckets.
    Simulates Prophet/ARIMA forecasts with confidence bounds.
    """

    def forecast(self, topic: str) -> TrendSeries:
        now = datetime.now(timezone.utc)
        # Generate 6 historical months
        historical: list[TrendPoint] = []
        base_val = 15.0 + (_deterministic_float(f"trend_base:{topic}") * 35.0)
        growth_factor = 1.0 + (_deterministic_float(f"trend_growth:{topic}") * 0.4 - 0.1)

        for i in range(6, 0, -1):
            ts = now - timedelta(days=i * 30)
            val = max(1.0, base_val * math.pow(growth_factor, -i / 6.0))
            historical.append(
                TrendPoint(
                    timestamp=ts,
                    value=round(val, 2),
                    lower_bound=round(val * 0.9, 2),
                    upper_bound=round(val * 1.1, 2),
                )
            )

        # Generate 6 forecast months
        forecast: list[TrendPoint] = []
        current_val = base_val
        for i in range(1, 7):
            ts = now + timedelta(days=i * 30)
            current_val *= (1.0 + (growth_factor - 1.0) / 6.0)
            forecast.append(
                TrendPoint(
                    timestamp=ts,
                    value=round(current_val, 2),
                    lower_bound=round(current_val * 0.82, 2),
                    upper_bound=round(current_val * 1.22, 2),
                )
            )

        annual_growth = round((growth_factor - 1.0) * 100, 1)
        is_emerging = annual_growth > 18.0

        return TrendSeries(
            topic=topic,
            historical_points=historical,
            forecast_points=forecast,
            growth_rate=annual_growth,
            is_emerging=is_emerging,
        )


class StubImpactPredictor(ImpactProtocol):
    """
    Mock citation impact predictor.
    Simulates gradient-boosted regression on SPECTER embeddings + author/venue features.
    """

    def predict(self, paper: Paper) -> float:
        author_factor = len(paper.authors) * 1.5
        prior_citations = float(paper.citation_count)
        pseudo_boost = _deterministic_float(f"impact:{paper.id}") * 25.0

        predicted = max(0.0, prior_citations * 1.35 + author_factor + pseudo_boost)
        return round(predicted, 1)
