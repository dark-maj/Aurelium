from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class BaseContract(BaseModel):
    """Base Pydantic v2 model configuration for Aurelium contracts."""
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={datetime: lambda dt: dt.isoformat()},
    )


class Paper(BaseContract):
    """
    Representation of an academic research paper.
    Preserves publication date and external identifiers across the entire pipeline.
    """
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Internal unique identifier for the paper.",
    )
    title: str = Field(..., description="Paper title.")
    abstract: str = Field(..., description="Abstract text.")
    authors: list[str] = Field(default_factory=list, description="Author names in citation order.")
    venue: Optional[str] = Field(None, description="Publication venue or journal/conference name.")
    year: Optional[int] = Field(None, description="Year of publication.")
    published_date: Optional[datetime] = Field(
        None,
        description="Precise publication date (critical to prevent future-leakage in novelty & forecasting).",
    )
    doi: Optional[str] = Field(None, description="Digital Object Identifier.")
    arxiv_id: Optional[str] = Field(None, description="arXiv identifier (e.g., '2305.18290').")
    semantic_scholar_id: Optional[str] = Field(
        None,
        description="Semantic Scholar identifier (CorpusId or Paper SHA).",
    )
    citation_count: int = Field(default=0, ge=0, description="Observed citation count.")
    fields_of_study: list[str] = Field(
        default_factory=list,
        description="Identified academic fields of study (e.g., 'Computer Science').",
    )
    url: Optional[str] = Field(None, description="Canonical URL to preprint or publication.")
    embedding: Optional[list[float]] = Field(
        None,
        description="768-dimensional normalized SPECTER/SciBERT vector.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary extensible metadata from ingestion sources.",
    )


class Chunk(BaseContract):
    """
    Indexed segment of a paper with character boundaries for verifiable citation grounding.
    """
    chunk_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the chunk.",
    )
    paper_id: str = Field(..., description="Foreign key referencing Paper.id.")
    section: Optional[str] = Field(
        None,
        description="Section header where the chunk originates (e.g., 'Introduction', 'Methods').",
    )
    text: str = Field(..., description="Extracted text content of the chunk.")
    char_start: int = Field(..., ge=0, description="Start character index in the parent document.")
    char_end: int = Field(..., ge=0, description="End character index in the parent document.")
    token_count: Optional[int] = Field(None, ge=0, description="Token count if tokenized.")
    embedding: Optional[list[float]] = Field(
        None,
        description="Dense vector embedding for vector search.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata (e.g., page number, bounding box).",
    )


class Summary(BaseContract):
    """
    Structured analytical summary of a research paper produced by LLM agents.
    Every factual claim must be grounded in paper/chunk citations.
    """
    paper_id: str = Field(..., description="Identifier of the paper summarized.")
    tl_dr: str = Field(..., description="Concise 1-2 sentence executive summary.")
    key_contributions: list[str] = Field(
        default_factory=list,
        description="Core theoretical or empirical contributions.",
    )
    methodology: str = Field(..., description="High-level description of proposed techniques.")
    limitations: list[str] = Field(
        default_factory=list,
        description="Identified limitations or constraints.",
    )
    citations: list[str] = Field(
        default_factory=list,
        description="Chunk IDs or reference markers grounding the summary claims.",
    )
    novelty_assessment: Optional[str] = Field(
        None,
        description="Qualitative assessment informed by ML novelty score.",
    )
    predicted_impact_assessment: Optional[str] = Field(
        None,
        description="Commentary informed by ML citation prediction.",
    )


class ComparisonCell(BaseContract):
    """
    Single cell in a multi-paper comparison matrix with citation traceability.
    """
    paper_id: str = Field(..., description="Identifier of the paper being compared.")
    dimension: str = Field(
        ...,
        description="Dimension or metric evaluated (e.g., 'Methodology', 'Dataset', 'Compute').",
    )
    value: str = Field(..., description="Extracted or synthesized value for the dimension.")
    citation_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs providing evidence for the value.",
    )


class ComparisonTable(BaseContract):
    """
    Multi-paper comparative matrix evaluating papers across consistent criteria.
    """
    topic: str = Field(..., description="Research topic or question governing the comparison.")
    dimensions: list[str] = Field(
        ...,
        description="Ordered list of evaluated dimensions.",
    )
    paper_ids: list[str] = Field(
        ...,
        description="List of compared paper IDs.",
    )
    cells: list[ComparisonCell] = Field(
        ...,
        description="Flat or tabular cell values connecting papers to dimensions.",
    )
    synthesis: str = Field(
        ...,
        description="Comparative synthesis highlighting trade-offs, Pareto frontier, and consensus.",
    )


class GapReport(BaseContract):
    """
    Systematic report on literature gaps, contradictions, and under-explored avenues.
    """
    topic: str = Field(..., description="Domain or topic investigated for gaps.")
    analyzed_paper_ids: list[str] = Field(
        ...,
        description="IDs of papers analyzed to deduce literature gaps.",
    )
    unaddressed_problems: list[str] = Field(
        default_factory=list,
        description="Explicit open problems recognized in the analyzed literature.",
    )
    contradictions_or_conflicts: list[str] = Field(
        default_factory=list,
        description="Conflicting empirical results or theoretical arguments between papers.",
    )
    under_explored_avenues: list[str] = Field(
        default_factory=list,
        description="High-potential hypotheses or paradigms currently lacking coverage.",
    )
    methodological_bottlenecks: list[str] = Field(
        default_factory=list,
        description="Persistent tooling, evaluation, dataset, or compute bottlenecks.",
    )
    synthesis: str = Field(
        ...,
        description="Synthesis grounding the gaps and prioritizing opportunities.",
    )
    citation_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs grounding the identified gaps.",
    )


class Hypothesis(BaseContract):
    """
    Novel, falsifiable research hypothesis grounded in identified gaps.
    """
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the hypothesis.",
    )
    title: str = Field(..., description="Descriptive title for the hypothesis.")
    statement: str = Field(..., description="Precise, falsifiable hypothesis statement.")
    rationale: str = Field(
        ...,
        description="Theoretical and empirical motivation derived from retrieved papers.",
    )
    grounded_paper_ids: list[str] = Field(
        default_factory=list,
        description="Papers providing foundational premises.",
    )
    grounded_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Specific chunks cited as evidence.",
    )
    target_gap: str = Field(..., description="Specific gap or conflict targeted.")
    proposed_methodology: str = Field(
        ...,
        description="Suggested experimental architecture or protocol to test the hypothesis.",
    )
    expected_outcomes: list[str] = Field(
        default_factory=list,
        description="Measurable validation criteria.",
    )
    testability_and_risks: str = Field(
        ...,
        description="Assessment of feasibility, confounders, and failure modes.",
    )
    confidence_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Subjective confidence rating (0.0 to 1.0).",
    )


class NoveltyScore(BaseContract):
    """
    Quantitative novelty assessment for a paper relative to historical literature.
    """
    paper_id: str = Field(..., description="Evaluated paper ID.")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Novelty score in [0.0, 1.0] (1 - mean cosine similarity to prior k-NN).",
    )
    nearest_prior_distance: Optional[float] = Field(
        None,
        description="Distance to nearest prior neighbor in embedding space.",
    )
    percentile: Optional[float] = Field(
        None,
        description="Percentile ranking against the historical distribution of novelty.",
    )
    prior_papers_count: Optional[int] = Field(
        None,
        description="Number of prior papers against which the neighborhood was evaluated.",
    )


class TrendPoint(BaseContract):
    """Single time-series point in a topic trajectory."""
    timestamp: datetime = Field(..., description="Timestamp of the interval bucket.")
    value: float = Field(..., description="Publication volume, velocity, or interest score.")
    lower_bound: Optional[float] = Field(None, description="Lower confidence interval.")
    upper_bound: Optional[float] = Field(None, description="Upper confidence interval.")


class TrendSeries(BaseContract):
    """
    Historical trajectory and future forecast for a topic or semantic cluster.
    """
    topic: str = Field(..., description="Name of the topic or cluster.")
    historical_points: list[TrendPoint] = Field(
        default_factory=list,
        description="Observed historical trajectory.",
    )
    forecast_points: list[TrendPoint] = Field(
        default_factory=list,
        description="Future forecast points (e.g., Prophet / ARIMA model).",
    )
    growth_rate: float = Field(
        default=0.0,
        description="Estimated annual compound growth percentage.",
    )
    is_emerging: bool = Field(
        default=False,
        description="Flag indicating if the trajectory exceeds the threshold for an emerging trend.",
    )


class ResearchResult(BaseContract):
    """
    Comprehensive result package produced by the Aurelium AI Copilot.
    """
    query: str = Field(..., description="User query or research topic.")
    ranked_papers: list[Paper] = Field(
        default_factory=list,
        description="Retrieved papers sorted by relevance.",
    )
    ranking_scores: list[float] = Field(
        default_factory=list,
        description="Relevance scores corresponding to ranked_papers.",
    )
    summaries: list[Summary] = Field(
        default_factory=list,
        description="Grounded analytical summaries for top papers.",
    )
    comparison_table: Optional[ComparisonTable] = Field(
        None,
        description="Comparative table across key dimensions.",
    )
    gap_report: Optional[GapReport] = Field(
        None,
        description="Identified open gaps and contradictions in the literature.",
    )
    hypotheses: list[Hypothesis] = Field(
        default_factory=list,
        description="Generated novel research hypotheses.",
    )
    novelty_scores: list[NoveltyScore] = Field(
        default_factory=list,
        description="Novelty evaluations for candidate papers.",
    )
    trend_series: Optional[TrendSeries] = Field(
        None,
        description="Topic forecast trajectory.",
    )
    synthesis: str = Field(
        default="",
        description="Comprehensive narrative synthesis answering the user query.",
    )
