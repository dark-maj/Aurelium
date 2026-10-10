from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field

from contracts.schemas.models import Paper


class RetrievalFilters(BaseModel):
    """Optional metadata filters applied to both dense and sparse retrieval stages."""
    year_min: Optional[int] = Field(None, description="Earliest publication year (inclusive).")
    year_max: Optional[int] = Field(None, description="Latest publication year (inclusive).")
    fields_of_study: Optional[list[str]] = Field(None, description="List of allowed disciplines/fields.")
    min_citations: Optional[int] = Field(None, description="Minimum citation count.")
    similarity_threshold: Optional[float] = Field(
        None,
        description="Override threshold below which retrieval is marked as weak.",
    )


class StageScores(BaseModel):
    """Scores recorded at each retrieval and ranking stage for auditability."""
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rrf_score: float = 0.0
    cross_encoder_score: Optional[float] = None
    ranker_score: Optional[float] = None


class ScoredChunk(BaseModel):
    """
    Hydrated retrieval result containing all chunk text boundaries,
    attached paper metadata, and multi-stage scoring.
    """
    chunk_id: str
    paper_id: str
    section: Optional[str] = None
    text: str
    char_start: int
    char_end: int
    paper: Optional[Paper] = None
    scores: StageScores = Field(default_factory=StageScores)
    final_score: float = 0.0


class RetrievalResult(BaseModel):
    """
    Consolidated response returned by the retrieve() interface.
    Contains hydrated chunks, weak-retrieval diagnostic signal, and latency breakdown.
    """
    query: str
    chunks: list[ScoredChunk] = Field(default_factory=list)
    is_weak: bool = Field(
        default=False,
        description="True if top candidate scores fall below confidence threshold or no chunks found.",
    )
    weak_reason: Optional[str] = Field(
        None,
        description="Diagnostic explanation when is_weak is True.",
    )
    latencies_ms: dict[str, float] = Field(
        default_factory=dict,
        description="Latency breakdown in milliseconds per stage (dense, sparse, fusion, hydration, ranker).",
    )
