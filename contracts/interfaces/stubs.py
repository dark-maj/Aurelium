from __future__ import annotations

import hashlib
import math
from datetime import datetime, timedelta, timezone
from typing import Sequence

from contracts.schemas.models import Paper, TrendPoint, TrendSeries
from contracts.interfaces.protocols import (
    RankerProtocol,
    NoveltyProtocol,
    ImpactProtocol,
    TrendProtocol,
)


def _deterministic_hash_float(seed: str) -> float:
    """Generate a reproducible float in [0.0, 1.0) given a seed string."""
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


class StubRanker(RankerProtocol):
    """
    Deterministic stub implementation of RankerProtocol.
    Returns reproducible synthetic relevance scores using word overlap and hash dispersion.
    """

    def rank(self, query: str, papers: Sequence[Paper]) -> list[float]:
        query_words = set(query.lower().split())
        scores: list[float] = []

        for paper in papers:
            title_words = set(paper.title.lower().split())
            abstract_words = set(paper.abstract.lower().split())
            overlap = len(query_words & (title_words | abstract_words))

            base_overlap_score = min(0.6, overlap * 0.15)
            pseudo_semantic = _deterministic_hash_float(f"{query}:{paper.id}") * 0.4
            score = round(min(1.0, max(0.05, base_overlap_score + pseudo_semantic)), 4)
            scores.append(score)

        return scores


class StubNoveltyScorer(NoveltyProtocol):
    """
    Deterministic stub implementation of NoveltyProtocol.
    Returns synthetic novelty score in [0.15, 0.92] based on paper ID and title hash.
    """

    def score(self, paper: Paper) -> float:
        pseudo_dist = _deterministic_hash_float(f"novelty:{paper.id}:{paper.title}")
        return round(0.15 + (pseudo_dist * 0.77), 4)


class StubImpactPredictor(ImpactProtocol):
    """
    Deterministic stub implementation of ImpactProtocol.
    Returns synthetic citation prediction based on author count and existing citations.
    """

    def predict(self, paper: Paper) -> float:
        author_factor = len(paper.authors) * 1.5
        prior_citations = float(paper.citation_count)
        pseudo_boost = _deterministic_hash_float(f"impact:{paper.id}") * 25.0
        predicted = max(0.0, prior_citations * 1.35 + author_factor + pseudo_boost)
        return round(predicted, 1)


class StubTrendForecaster(TrendProtocol):
    """
    Deterministic stub implementation of TrendProtocol.
    Returns a synthetic TrendSeries with 6 historical and 6 forecasted monthly points.
    """

    def forecast(self, topic: str) -> TrendSeries:
        # Use a fixed reference base date for complete test determinism
        base_date = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        base_val = 20.0 + (_deterministic_hash_float(f"trend_base:{topic}") * 30.0)
        growth_factor = 1.0 + (_deterministic_hash_float(f"trend_growth:{topic}") * 0.4 - 0.1)

        historical: list[TrendPoint] = []
        for i in range(6, 0, -1):
            ts = base_date - timedelta(days=i * 30)
            val = max(1.0, base_val * math.pow(growth_factor, -i / 6.0))
            historical.append(
                TrendPoint(
                    timestamp=ts,
                    value=round(val, 2),
                    lower_bound=round(val * 0.9, 2),
                    upper_bound=round(val * 1.1, 2),
                )
            )

        forecast: list[TrendPoint] = []
        curr = base_val
        for i in range(1, 7):
            ts = base_date + timedelta(days=i * 30)
            curr *= (1.0 + (growth_factor - 1.0) / 6.0)
            forecast.append(
                TrendPoint(
                    timestamp=ts,
                    value=round(curr, 2),
                    lower_bound=round(curr * 0.82, 2),
                    upper_bound=round(curr * 1.22, 2),
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
