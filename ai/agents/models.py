from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

from contracts.schemas.models import Summary as ContractSummary


class ClaimCitation(BaseModel):
    """
    Individual evidence-backed claim grounded in specific document chunk IDs.
    If a dimension was not reported in the paper, statement is 'not reported' with chunk_ids: [].
    """
    statement: str = Field(..., description="Extracted claim, factual statement, or 'not reported'.")
    chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs providing direct citation grounding for this statement.",
    )
    verification_status: Optional[str] = Field(
        default=None,
        description="Verification outcome: 'supported', 'partially_supported', or 'unsupported'.",
    )
    verification_reason: Optional[str] = Field(
        default=None,
        description="Justification or critique provided by the verification pass.",
    )


class PaperSummary(BaseModel):
    """
    Structured analytical summary across 5 mandatory scientific research dimensions.
    Every dimension contains a list of verified claims citing chunk IDs.
    """
    paper_id: str = Field(..., description="Target paper ID.")
    problem: list[ClaimCitation] = Field(
        default_factory=list,
        description="Core problem, research question, or theoretical challenges.",
    )
    method: list[ClaimCitation] = Field(
        default_factory=list,
        description="Proposed methodology, algorithm, architecture, or technique.",
    )
    datasets: list[ClaimCitation] = Field(
        default_factory=list,
        description="Datasets, benchmarks, or experimental corpora used for evaluation.",
    )
    results: list[ClaimCitation] = Field(
        default_factory=list,
        description="Key quantitative and qualitative empirical findings.",
    )
    limitations: list[ClaimCitation] = Field(
        default_factory=list,
        description="Stated assumptions, constraints, failure modes, or compute bounds.",
    )
    verification_log: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Auditing log of all verification evaluations executed for this paper.",
    )

    def to_contract_summary(self) -> ContractSummary:
        """Converts to the contracts.schemas.models.Summary representation."""
        # Synthesize tl_dr from first problem and result statements
        p_stmt = self.problem[0].statement if self.problem else "Not reported"
        r_stmt = self.results[0].statement if self.results else "Not reported"
        tl_dr = f"Problem: {p_stmt} Findings: {r_stmt}"

        # Contributions from method and results
        contributions = [
            c.statement for c in self.method if c.statement.lower() != "not reported"
        ]

        # Methodology text
        method_stmts = [
            c.statement for c in self.method if c.statement.lower() != "not reported"
        ]
        methodology = "; ".join(method_stmts) if method_stmts else "not reported"

        # Limitations list
        limits = [
            c.statement for c in self.limitations if c.statement.lower() != "not reported"
        ]

        # Collect all unique chunk citation IDs
        all_citations: set[str] = set()
        for field in [self.problem, self.method, self.datasets, self.results, self.limitations]:
            for claim in field:
                all_citations.update(claim.chunk_ids)

        return ContractSummary(
            paper_id=self.paper_id,
            tl_dr=tl_dr,
            key_contributions=contributions,
            methodology=methodology,
            limitations=limits,
            citations=sorted(list(all_citations)),
        )


# Alias Summary to PaperSummary for convenience
Summary = PaperSummary


class ClaimVerificationResult(BaseModel):
    """Individual verdict produced by the verifier LLM pass."""
    claim_index: int = Field(..., description="Index corresponding to the evaluated candidate claim.")
    dimension: str = Field(..., description="The dimension name (e.g. 'results', 'method').")
    statement: str = Field(..., description="The candidate statement that was verified.")
    status: Literal["supported", "partially_supported", "unsupported"] = Field(
        ...,
        description="Verification classification.",
    )
    reason: str = Field(..., description="Short explanation justification for the verdict.")


class BatchVerificationResponse(BaseModel):
    """Batched response returned by the verification pass."""
    verdicts: list[ClaimVerificationResult] = Field(
        default_factory=list,
        description="Evaluated verdicts for each claim submitted in the batch.",
    )


class MultiPaperSummaryResult(BaseModel):
    """Collection of paper summaries along with an overarching cross-paper synthesis."""
    summaries: list[PaperSummary] = Field(
        default_factory=list,
        description="List of verified structured summaries for each paper.",
    )
    cross_paper_overview: str = Field(
        ...,
        description="High-density cross-paper research synthesis citing paper IDs in brackets.",
    )
