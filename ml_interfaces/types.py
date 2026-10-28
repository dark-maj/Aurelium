from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class TrendPoint(BaseModel):
    """A single forecasted time point."""
    timestamp: datetime = Field(..., description="Timestamp of the forecast interval (e.g. monthly).")
    value: float = Field(..., description="Forecasted metric value (e.g., publication volume or interest).")
    lower_bound: Optional[float] = Field(None, description="Lower confidence interval.")
    upper_bound: Optional[float] = Field(None, description="Upper confidence interval.")


class TrendSeries(BaseModel):
    """Time-series trajectory for a topic or cluster."""
    topic: str = Field(..., description="Analyzed topic or cluster name.")
    historical_points: list[TrendPoint] = Field(
        default_factory=list,
        description="Historical counts preceding the forecast horizon."
    )
    forecast_points: list[TrendPoint] = Field(
        default_factory=list,
        description="Forward-looking forecast points (e.g., next 6-12 months)."
    )
    growth_rate: float = Field(
        default=0.0,
        description="Estimated annual or period growth percentage (e.g., 0.35 = 35% growth)."
    )
    is_emerging: bool = Field(
        default=False,
        description="True if the trajectory exceeds the novelty/velocity threshold for emerging topics."
    )


class RankedScore(BaseModel):
    """Relevance ranking result for a single paper."""
    paper_id: str
    score: float
