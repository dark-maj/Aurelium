from __future__ import annotations

from typing import List, Optional, TypedDict, Annotated
import operator
from pydantic import BaseModel, Field

from schemas.models import Paper, Summary, ComparisonTable, GapReport, Hypothesis
from retrieval.retriever import EnrichedPaperResult
from ml_interfaces.types import TrendSeries


class ResearchState(BaseModel):
    """LangGraph State representation for the AI Research Copilot workflow."""
    query: str = Field(..., description="User research question or topic.")
    task_type: str = Field(
        default="deep_dive",
        description="Type of copilot task: 'deep_dive', 'literature_review', 'gap_analysis', or 'hypothesis_generation'."
    )
    retrieved_papers: List[EnrichedPaperResult] = Field(
        default_factory=list,
        description="Top retrieved papers with attached ML signals."
    )
    topic_trend: Optional[TrendSeries] = Field(
        None,
        description="Forecasted trend trajectory for the overarching topic."
    )
    summaries: List[Summary] = Field(
        default_factory=list,
        description="Extracted paper summaries."
    )
    comparison_table: Optional[ComparisonTable] = Field(
        None,
        description="Comparative analysis across papers."
    )
    gap_report: Optional[GapReport] = Field(
        None,
        description="Literature gaps and open problems deduced by the agent."
    )
    hypotheses: List[Hypothesis] = Field(
        default_factory=list,
        description="Generated research hypotheses targeting the identified gaps."
    )
    final_response: str = Field(
        default="",
        description="Synthesized natural language answer with grounded citations."
    )
