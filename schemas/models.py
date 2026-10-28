from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


class BaseSchema(BaseModel):
    """Base schema configuration for Aurelium."""
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={datetime: lambda dt: dt.isoformat()}
    )


class Paper(BaseSchema):
    """
    Representation of an academic research paper ingested from arXiv, Semantic Scholar,
    PubMed, or direct PDF ingestion.
    """
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the paper (e.g., DOI, arXiv ID, or UUID)."
    )
    title: str = Field(..., description="Paper title.")
    abstract: str = Field(..., description="Abstract text.")
    authors: list[str] = Field(default_factory=list, description="List of author names.")
    venue: Optional[str] = Field(None, description="Publication venue or journal/conference name.")
    year: Optional[int] = Field(None, description="Year of publication.")
    published_date: Optional[datetime] = Field(None, description="Full publication date if available.")
    doi: Optional[str] = Field(None, description="Digital Object Identifier.")
    arxiv_id: Optional[str] = Field(None, description="arXiv ID (e.g., 2305.18290).")
    fields_of_study: list[str] = Field(
        default_factory=list,
        description="Relevant disciplines/topics (e.g., 'Computer Science', 'Machine Learning')."
    )
    citation_count: int = Field(default=0, ge=0, description="Current known citation count.")
    url: Optional[str] = Field(None, description="URL pointing to paper landing page or PDF.")
    embedding: Optional[list[float]] = Field(
        None,
        description="Dense vector embedding (e.g., 768-dim SPECTER/SciBERT embedding)."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata (e.g., source repository, licensing, raw attributes)."
    )


class Chunk(BaseSchema):
    """
    Segment of a paper (abstract, introduction, section) used for vector retrieval in pgvector.
    """
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the chunk."
    )
    paper_id: str = Field(..., description="Foreign key referencing Paper.id.")
    chunk_index: int = Field(..., ge=0, description="0-indexed position within the document.")
    content: str = Field(..., description="Text content of the chunk.")
    section: Optional[str] = Field(
        None,
        description="Section header where the chunk originates (e.g., 'Abstract', 'Methodology')."
    )
    token_count: Optional[int] = Field(None, ge=0, description="Approximate token count.")
    embedding: Optional[list[float]] = Field(
        None,
        description="Dense vector embedding for retrieval."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional chunk metadata (e.g., page number, bounding box)."
    )


class Summary(BaseSchema):
    """
    Structured analytical summary of a research paper produced by LLM agents,
    augmented with ML insights.
    """
    paper_id: str = Field(..., description="Identifier of the summarized paper.")
    tl_dr: str = Field(..., description="1-2 sentence executive takeaway.")
    key_contributions: list[str] = Field(
        default_factory=list,
        description="Core technical or theoretical contributions."
    )
    methodology: str = Field(..., description="High-level description of proposed techniques.")
    limitations: list[str] = Field(
        default_factory=list,
        description="Author-acknowledged or critically inferred limitations."
    )
    novelty_assessment: Optional[str] = Field(
        None,
        description="Qualitative assessment conditioned on the ML novelty score."
    )
    predicted_impact_assessment: Optional[str] = Field(
        None,
        description="Contextual commentary on expected citation velocity and influence."
    )


class ComparisonRow(BaseSchema):
    """Row in a structured multi-paper comparison matrix."""
    paper_id: str = Field(..., description="Paper ID.")
    paper_title: str = Field(..., description="Title of the paper.")
    cells: dict[str, str] = Field(
        default_factory=dict,
        description="Key-value mapping of criterion -> analysis text (e.g. {'Method': 'Diffusion', 'Compute': '8x A100'})."
    )


class ComparisonTable(BaseSchema):
    """
    Multi-paper comparative synthesis evaluating papers across uniform criteria.
    """
    topic: str = Field(..., description="Topic or research question uniting the compared papers.")
    dimension_names: list[str] = Field(
        ...,
        description="Attribute columns compared (e.g., ['Methodology', 'Dataset', 'Compute Cost', 'Strengths', 'Weaknesses'])."
    )
    rows: list[ComparisonRow] = Field(..., description="Evaluated rows for each paper.")
    synthesis: str = Field(
        ...,
        description="LLM comparative synthesis highlighting trade-offs, Pareto frontiers, and consensus."
    )


class GapReport(BaseSchema):
    """
    Detailed identification of literature gaps, contradictions, and under-explored avenues.
    """
    topic: str = Field(..., description="Subject domain of the gap analysis.")
    analyzed_paper_ids: list[str] = Field(
        ...,
        description="List of paper IDs ingested to deduce literature gaps."
    )
    unaddressed_problems: list[str] = Field(
        default_factory=list,
        description="Explicit open problems recognized in the literature."
    )
    contradictions_or_conflicts: list[str] = Field(
        default_factory=list,
        description="Empirical or theoretical discrepancies between cited works."
    )
    under_explored_avenues: list[str] = Field(
        default_factory=list,
        description="High-potential directions with little to no current coverage."
    )
    methodological_bottlenecks: list[str] = Field(
        default_factory=list,
        description="Common tool/dataset/architecture barriers holding back the domain."
    )
    synthesis: str = Field(
        ...,
        description="Holistic discussion grounding the gaps and prioritizing opportunities."
    )


class Hypothesis(BaseSchema):
    """
    Novel research hypothesis generated by LLM agents, grounded in literature gaps
    and calibrated against ML novelty and impact predictors.
    """
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the hypothesis."
    )
    title: str = Field(..., description="Short, descriptive title for the hypothesis.")
    statement: str = Field(
        ...,
        description="Precise, falsifiable hypothesis statement."
    )
    rationale: str = Field(
        ...,
        description="Theoretical and empirical motivation derived from retrieved papers."
    )
    grounded_paper_ids: list[str] = Field(
        ...,
        description="Papers providing the foundational premises or evidence."
    )
    target_gap: str = Field(
        ...,
        description="Reference to the specific gap or contradiction being addressed."
    )
    proposed_methodology: str = Field(
        ...,
        description="Suggested experimental design or architecture to validate/invalidate."
    )
    expected_outcomes: list[str] = Field(
        default_factory=list,
        description="Predicted measurable indicators of validation success."
    )
    testability_and_risks: str = Field(
        ...,
        description="Assessment of feasibility, confounders, and failure modes."
    )
    confidence_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Subjective confidence rating (0.0 - 1.0) calibrated by the agent."
    )
