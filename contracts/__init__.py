"""
contracts - Shared contract layer between ai/ and ml/ halves of Aurelium.
Contains domain schemas, typing.Protocol definitions, stubs, and OpenAPI specs.
"""

from contracts.schemas.models import (
    BaseContract,
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

__all__ = [
    "BaseContract",
    "Paper",
    "Chunk",
    "Summary",
    "Claim",
    "ComparisonCell",
    "ComparisonTable",
    "GapReport",
    "Hypothesis",
    "NoveltyScore",
    "TrendPoint",
    "TrendSeries",
    "ResearchResult",
    "RankerProtocol",
    "NoveltyProtocol",
    "ImpactProtocol",
    "TrendProtocol",
    "StubRanker",
    "StubNoveltyScorer",
    "StubImpactPredictor",
    "StubTrendForecaster",
]
