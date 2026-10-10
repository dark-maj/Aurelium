from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Sequence

try:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncEngine
except ImportError:
    def text(query: str) -> Any:  # type: ignore
        return query

    AsyncEngine = Any  # type: ignore

try:
    from ai.db.session import engine as default_engine
except ImportError:
    default_engine = None  # type: ignore

from contracts.schemas.models import Chunk
from ai.agents.cache import LLMCache, global_llm_cache
from ai.agents.llm import BaseLLMClient, GrokClient
from ai.agents.models import (
    BatchVerificationResponse,
    ClaimCitation,
    ClaimVerificationResult,
    MultiPaperSummaryResult,
    PaperSummary,
)
from ai.agents.prompt_loader import load_prompt

logger = logging.getLogger("aurelium.ai.agents.summarizer")


class SummarizationAgent:
    """
    Evidence-grounded research summarization agent.
    - Fetches document chunks from Neon Postgres.
    - Generates 5-dimension structured summaries (problem, method, datasets, results, limitations)
      via Grok structured JSON completions.
    - Runs a batched verifier pass against cited chunk text, removing unsupported claims
      and annotating partially supported claims.
    - Preserves 'not reported' statements when information is absent.
    - Manages concurrency across multiple papers.
    - Produces a cross-paper synthesis citing bracketed paper IDs.
    - Caches all LLM calls deterministically by prompt_version:model:sha256(payload).
    """

    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        db_engine: AsyncEngine | None = None,
        cache: LLMCache | None = None,
        concurrency_limit: int = 5,
    ):
        self.client = llm_client or GrokClient()
        self.engine = db_engine or default_engine
        self.cache = cache or global_llm_cache
        self.concurrency_limit = concurrency_limit

    async def fetch_chunks(self, paper_id: str) -> list[Chunk]:
        """Fetches all text chunks for a paper from Neon Postgres ordered by character offset."""
        if not self.engine:
            logger.warning("No database engine configured; returning empty chunks.")
            return []

        sql = text("""
            SELECT chunk_id, paper_id, section, text, char_start, char_end, token_count, metadata
            FROM chunks
            WHERE paper_id = :paper_id
            ORDER BY char_start ASC;
        """)

        try:
            async with self.engine.connect() as conn:
                result = await conn.execute(sql, {"paper_id": paper_id})
                rows = result.fetchall()

            chunks: list[Chunk] = []
            for row in rows:
                meta = row[7]
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = {}
                elif not isinstance(meta, dict):
                    meta = {}

                chunks.append(
                    Chunk(
                        chunk_id=str(row[0]),
                        paper_id=str(row[1]),
                        section=row[2],
                        text=str(row[3]),
                        char_start=int(row[4]),
                        char_end=int(row[5]),
                        token_count=int(row[6]) if row[6] is not None else None,
                        metadata=meta,
                    )
                )
            return chunks
        except Exception as exc:
            logger.error(f"Failed to fetch chunks from Neon for paper {paper_id}: {exc}")
            return []

    def _format_chunks_payload(self, paper_id: str, chunks: Sequence[Chunk]) -> str:
        """Formats chunks into an unambiguous context document with explicit chunk IDs."""
        formatted_chunks = []
        for c in chunks:
            sec_header = f" | Section: {c.section}" if c.section else ""
            formatted_chunks.append(
                f"[Chunk ID: {c.chunk_id}{sec_header}]\n{c.text.strip()}\n"
            )
        return (
            f"TARGET PAPER ID: {paper_id}\n\n"
            f"AVAILABLE PAPER CHUNKS ({len(chunks)} chunks):\n\n"
            + "\n--------------------\n".join(formatted_chunks)
        )

    async def _run_summarizer_llm(
        self,
        paper_id: str,
        chunks: Sequence[Chunk],
    ) -> PaperSummary:
        """Invokes Grok structured JSON completion or returns cached initial summary."""
        prompt_tmpl = load_prompt("summarizer")
        context_payload = self._format_chunks_payload(paper_id, chunks)
        user_prompt = (
            f"Analyze the following research paper chunks and extract the structured summary.\n\n"
            f"{context_payload}"
        )

        model_name = getattr(self.client, "model", "default_model")
        cache_key_payload = f"paper:{paper_id}\n{context_payload}"

        cached = self.cache.get(
            prompt_version=prompt_tmpl.version,
            model=model_name,
            input_payload=cache_key_payload,
            schema_model=PaperSummary,
        )
        if cached is not None:
            logger.debug(f"Cache HIT for paper summary: {paper_id}")
            return cached

        summary = await self.client.complete_json(
            prompt=user_prompt,
            schema_model=PaperSummary,
            system_prompt=prompt_tmpl.content,
        )
        # Guarantee paper_id is populated correctly
        summary.paper_id = paper_id

        self.cache.set(
            prompt_version=prompt_tmpl.version,
            model=model_name,
            input_payload=cache_key_payload,
            value=summary,
        )
        return summary

    async def _run_verifier_pass(
        self,
        paper_id: str,
        raw_summary: PaperSummary,
        chunks: Sequence[Chunk],
    ) -> PaperSummary:
        """
        Batched verification pass:
        Evaluates candidate claims against cited chunk text.
        Removes unsupported claims, annotates partially supported claims,
        and logs verification results.
        """
        chunk_map = {c.chunk_id: c.text for c in chunks}
        prompt_tmpl = load_prompt("verifier")
        model_name = getattr(self.client, "model", "default_model")

        # Collect claims to verify across all 5 dimensions
        # Each item: (dim_name, original_claim, cited_text_or_missing)
        candidates: list[tuple[str, ClaimCitation, str]] = []
        dimensions = ["problem", "method", "datasets", "results", "limitations"]

        for dim in dimensions:
            claims: list[ClaimCitation] = getattr(raw_summary, dim, [])
            for claim in claims:
                # "not reported" claims have no citations and are not evaluated
                if claim.statement.strip().lower() == "not reported" or not claim.chunk_ids:
                    continue

                # Find cited chunk texts
                found_texts = [
                    f"[Chunk {cid}]: {chunk_map[cid]}"
                    for cid in claim.chunk_ids
                    if cid in chunk_map
                ]
                if not found_texts:
                    # Grounding failure: cited chunks do not exist in document
                    candidates.append((dim, claim, "__NO_CHUNKS_FOUND__"))
                else:
                    candidates.append((dim, claim, "\n\n".join(found_texts)))

        verification_map: dict[tuple[str, str], tuple[str, str]] = {}
        verification_log: list[dict[str, Any]] = []

        # Immediately fail claims with non-existent chunks
        claims_for_llm: list[tuple[int, str, ClaimCitation, str]] = []
        for idx, (dim, claim, evidence) in enumerate(candidates):
            if evidence == "__NO_CHUNKS_FOUND__":
                status = "unsupported"
                reason = "Cited chunk IDs do not exist in the paper's indexed text."
                verification_map[(dim, claim.statement)] = (status, reason)
                verification_log.append({
                    "dimension": dim,
                    "statement": claim.statement,
                    "chunk_ids": claim.chunk_ids,
                    "status": status,
                    "reason": reason,
                })
            else:
                claims_for_llm.append((idx, dim, claim, evidence))

        if claims_for_llm:
            # Build batched verification prompt
            eval_items = []
            for idx, dim, claim, evidence in claims_for_llm:
                eval_items.append({
                    "claim_index": idx,
                    "dimension": dim,
                    "statement": claim.statement,
                    "cited_chunk_text": evidence,
                })

            eval_payload_json = json.dumps(eval_items, sort_keys=True)
            cached_verdict = self.cache.get(
                prompt_version=prompt_tmpl.version,
                model=model_name,
                input_payload=eval_payload_json,
                schema_model=BatchVerificationResponse,
            )

            if cached_verdict is not None:
                batch_response = cached_verdict
            else:
                verifier_user_prompt = (
                    f"Evaluate the factual accuracy and evidence grounding for the following "
                    f"candidate claims extracted from paper '{paper_id}'.\n\n"
                    f"Candidate Claims Batch:\n"
                    f"{eval_payload_json}\n\n"
                    f"Return a BatchVerificationResponse matching the schema with verdicts "
                    f"for each claim index."
                )

                batch_response = await self.client.complete_json(
                    prompt=verifier_user_prompt,
                    schema_model=BatchVerificationResponse,
                    system_prompt=prompt_tmpl.content,
                )
                self.cache.set(
                    prompt_version=prompt_tmpl.version,
                    model=model_name,
                    input_payload=eval_payload_json,
                    value=batch_response,
                )

            # Map verdicts back to claims
            verdict_by_idx = {v.claim_index: v for v in batch_response.verdicts}
            for idx, dim, claim, _ in claims_for_llm:
                v = verdict_by_idx.get(idx)
                if v:
                    status = v.status
                    reason = v.reason
                else:
                    status = "supported"
                    reason = "No adverse verdict reported by verifier."

                verification_map[(dim, claim.statement)] = (status, reason)
                verification_log.append({
                    "dimension": dim,
                    "statement": claim.statement,
                    "chunk_ids": claim.chunk_ids,
                    "status": status,
                    "reason": reason,
                })

        # Apply filtering and annotations across dimensions
        verified_data: dict[str, Any] = {
            "paper_id": paper_id,
            "verification_log": verification_log,
        }

        for dim in dimensions:
            raw_claims: list[ClaimCitation] = getattr(raw_summary, dim, [])
            filtered_claims: list[ClaimCitation] = []

            for claim in raw_claims:
                if claim.statement.strip().lower() == "not reported" or not claim.chunk_ids:
                    filtered_claims.append(
                        ClaimCitation(statement="not reported", chunk_ids=[])
                    )
                    continue

                verdict_info = verification_map.get((dim, claim.statement))
                if verdict_info:
                    status, reason = verdict_info
                    if status == "unsupported":
                        # Filter out unsupported claim
                        logger.info(
                            f"Discarding unsupported claim in {dim} for paper {paper_id}: "
                            f"'{claim.statement[:60]}...' (Reason: {reason})"
                        )
                        continue
                    elif status == "partially_supported":
                        claim.verification_status = "partially_supported"
                        claim.verification_reason = reason
                        filtered_claims.append(claim)
                    else:
                        claim.verification_status = "supported"
                        claim.verification_reason = reason
                        filtered_claims.append(claim)
                else:
                    filtered_claims.append(claim)

            # If all claims in this dimension were removed or none existed, fallback to 'not reported'
            if not filtered_claims:
                filtered_claims = [ClaimCitation(statement="not reported", chunk_ids=[])]

            verified_data[dim] = filtered_claims

        return PaperSummary(**verified_data)

    async def summarize_paper(
        self,
        paper_id: str,
        chunks: list[Chunk] | None = None,
    ) -> PaperSummary:
        """
        Summarizes a single paper:
        1. Fetches chunks from Neon (if not provided).
        2. Generates initial structured summary.
        3. Executes verification pass, filtering unsupported claims.
        """
        if chunks is None:
            chunks = await self.fetch_chunks(paper_id)

        if not chunks:
            logger.warning(f"No chunks available for paper {paper_id}; emitting empty summary.")
            return PaperSummary(
                paper_id=paper_id,
                problem=[ClaimCitation(statement="not reported", chunk_ids=[])],
                method=[ClaimCitation(statement="not reported", chunk_ids=[])],
                datasets=[ClaimCitation(statement="not reported", chunk_ids=[])],
                results=[ClaimCitation(statement="not reported", chunk_ids=[])],
                limitations=[ClaimCitation(statement="not reported", chunk_ids=[])],
                verification_log=[],
            )

        raw_summary = await self._run_summarizer_llm(paper_id, chunks)
        verified_summary = await self._run_verifier_pass(paper_id, raw_summary, chunks)
        return verified_summary

    async def generate_cross_paper_overview(
        self,
        summaries: Sequence[PaperSummary],
    ) -> str:
        """Synthesizes verified structured summaries into a high-density cross-paper overview."""
        if not summaries:
            return "No paper summaries provided for cross-paper synthesis."

        prompt_tmpl = load_prompt("overview")
        model_name = getattr(self.client, "model", "default_model")

        # Prepare summary digests for each paper
        paper_digests = []
        for s in summaries:
            p_stmts = "; ".join(c.statement for c in s.problem)
            m_stmts = "; ".join(c.statement for c in s.method)
            d_stmts = "; ".join(c.statement for c in s.datasets)
            r_stmts = "; ".join(c.statement for c in s.results)
            l_stmts = "; ".join(c.statement for c in s.limitations)
            digest = (
                f"Paper [{s.paper_id}]:\n"
                f"  - Problem: {p_stmts}\n"
                f"  - Method: {m_stmts}\n"
                f"  - Datasets: {d_stmts}\n"
                f"  - Results: {r_stmts}\n"
                f"  - Limitations: {l_stmts}\n"
            )
            paper_digests.append(digest)

        overview_input = (
            f"Synthesize the following paper summaries into a coherent cross-paper research overview.\n"
            f"Cite paper IDs in brackets (e.g. [{summaries[0].paper_id}]).\n\n"
            + "\n".join(paper_digests)
        )

        cache_payload = json.dumps(
            [s.model_dump(exclude={"verification_log"}) for s in summaries],
            sort_keys=True,
        )
        cached_str = self.cache.get_str(
            prompt_version=prompt_tmpl.version,
            model=model_name,
            input_payload=cache_payload,
        )
        if cached_str is not None:
            logger.debug("Cache HIT for cross-paper overview")
            return cached_str

        overview_text = await self.client.complete(
            prompt=overview_input,
            system_prompt=prompt_tmpl.content,
            temperature=0.2,
        )
        self.cache.set(
            prompt_version=prompt_tmpl.version,
            model=model_name,
            input_payload=cache_payload,
            value=overview_text,
        )
        return overview_text

    async def summarize_papers(
        self,
        paper_ids: Sequence[str],
        chunks_by_paper: dict[str, list[Chunk]] | None = None,
        max_concurrency: int | None = None,
    ) -> MultiPaperSummaryResult:
        """
        Concurrently summarizes a batch of papers with concurrency limiting,
        then synthesizes a cross-paper overview.
        """
        if not paper_ids:
            return MultiPaperSummaryResult(
                summaries=[],
                cross_paper_overview="No papers provided.",
            )

        concurrency = max_concurrency or self.concurrency_limit
        sem = asyncio.Semaphore(concurrency)

        async def _summarize_single(pid: str) -> PaperSummary:
            async with sem:
                chunks = chunks_by_paper.get(pid) if chunks_by_paper else None
                return await self.summarize_paper(pid, chunks=chunks)

        tasks = [_summarize_single(pid) for pid in paper_ids]
        summaries = await asyncio.gather(*tasks)

        overview = await self.generate_cross_paper_overview(summaries)
        return MultiPaperSummaryResult(
            summaries=list(summaries),
            cross_paper_overview=overview,
        )
