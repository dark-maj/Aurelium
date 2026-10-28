from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from typing import List, Optional

from schemas.models import Summary, ComparisonTable, GapReport, Hypothesis
from agents.state import ResearchState
from agents.graph import create_research_graph
from retrieval.retriever import HybridRetriever, EnrichedPaperResult
from ml_interfaces.protocols import TrendProtocol
from ml_interfaces.types import TrendSeries
from api.deps import get_retriever, get_trend_forecaster

router = APIRouter(prefix="/copilot", tags=["AI Copilot & Agents"])


class ResearchQueryRequest(BaseModel):
    query: str = Field(..., example="direct preference optimization vs reinforcement learning from human feedback")
    task_type: str = Field(default="deep_dive", example="deep_dive")


class ResearchQueryResponse(BaseModel):
    query: str
    final_response: str
    retrieved_papers: List[EnrichedPaperResult]
    topic_trend: Optional[TrendSeries]
    summaries: List[Summary]
    comparison_table: Optional[ComparisonTable]
    gap_report: Optional[GapReport]
    hypotheses: List[Hypothesis]


@router.post("/research", response_model=ResearchQueryResponse)
async def run_research_copilot(
    payload: ResearchQueryRequest,
    retriever: HybridRetriever = Depends(get_retriever),
    trend_forecaster: TrendProtocol = Depends(get_trend_forecaster),
):
    graph = create_research_graph(retriever=retriever, trend_forecaster=trend_forecaster)
    initial_state = ResearchState(query=payload.query, task_type=payload.task_type)
    final_state_dict = graph.invoke(initial_state)

    return ResearchQueryResponse(
        query=payload.query,
        final_response=final_state_dict["final_response"],
        retrieved_papers=final_state_dict["retrieved_papers"],
        topic_trend=final_state_dict.get("topic_trend"),
        summaries=final_state_dict.get("summaries", []),
        comparison_table=final_state_dict.get("comparison_table"),
        gap_report=final_state_dict.get("gap_report"),
        hypotheses=final_state_dict.get("hypotheses", []),
    )
