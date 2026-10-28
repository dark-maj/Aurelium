from __future__ import annotations

import hashlib
import io
import logging
import re
from typing import Any
import httpx

from contracts.schemas.models import Chunk, Paper

logger = logging.getLogger("aurelium.ai.ingestion.chunker")

SECTION_PATTERN = re.compile(
    r"\n(?:\d+\.?\s+)?(Abstract|Introduction|Related\s+Work|Background|Methods?|Methodology|Experiments?|Results|Discussion|Conclusions?)\b",
    re.IGNORECASE,
)


def make_chunk_id(paper_id: str, suffix: str) -> str:
    """Generates a deterministic unique chunk ID."""
    raw = f"{paper_id}:{suffix}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{paper_id}_{suffix}_{digest}"


def create_abstract_chunk(paper: Paper) -> Chunk:
    """
    Creates the default abstract-level chunk.
    Always includes the paper title in the chunk text to preserve semantic context.
    """
    text = f"Title: {paper.title}\n\nAbstract: {paper.abstract}"
    return Chunk(
        chunk_id=make_chunk_id(paper.id, "abs"),
        paper_id=paper.id,
        section="Abstract",
        text=text,
        char_start=0,
        char_end=len(text),
        token_count=len(text.split()),
        metadata={
            "paper_id": paper.id,
            "title": paper.title,
            "year": paper.year,
            "fields_of_study": paper.fields_of_study,
        },
    )


class PaperChunker:
    """
    Generates chunks for ingested papers.
    Uses abstract-level chunks by default.
    Supports optional section-aware chunking for full-text PDFs with graceful fallback.
    """

    def __init__(self, enable_full_text: bool = False, pdf_timeout: float = 15.0):
        self.enable_full_text = enable_full_text
        self.pdf_timeout = pdf_timeout

    async def _fetch_and_parse_pdf(self, pdf_url: str) -> str | None:
        """Fetches and extracts raw text from PDF URL if accessible."""
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=self.pdf_timeout) as client:
                resp = await client.get(pdf_url)
                if resp.status_code != 200:
                    return None
                pdf_bytes = resp.content

            # Attempt extraction via pypdf if installed
            try:
                import pypdf  # type: ignore
                reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
                text_pages = [page.extract_text() or "" for page in reader.pages]
                full_text = "\n".join(text_pages).strip()
                if len(full_text.split()) > 200:
                    return full_text
            except ImportError:
                logger.debug("pypdf not available for full-text extraction; falling back to abstract")
                return None

        except Exception as exc:
            logger.debug(f"Failed to fetch or extract PDF at {pdf_url}: {exc}")
            return None

        return None

    async def chunk_paper(self, paper: Paper) -> list[Chunk]:
        """
        Chunks a paper into one or more Chunk objects.
        If full text parsing fails or is disabled, falls back to abstract-level chunk.
        """
        pdf_url = paper.metadata.get("pdf_url")

        if self.enable_full_text and pdf_url:
            full_text = await self._fetch_and_parse_pdf(pdf_url)
            if full_text:
                section_chunks = self._chunk_sections(paper, full_text)
                if section_chunks:
                    return section_chunks

        # Default / Fallback: abstract chunk
        return [create_abstract_chunk(paper)]

    def _chunk_sections(self, paper: Paper, full_text: str) -> list[Chunk]:
        """Splits full text into section-aware chunks."""
        matches = list(SECTION_PATTERN.finditer(full_text))
        if not matches:
            return []

        chunks: list[Chunk] = []
        for i, match in enumerate(matches):
            sec_name = match.group(1).title()
            start_pos = match.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)

            body = full_text[start_pos:end_pos].strip()
            # Skip empty or trivial section fragments
            if len(body.split()) < 30:
                continue

            chunk_text = f"Title: {paper.title}\nSection: {sec_name}\n\n{body}"
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(paper.id, f"sec_{i}"),
                    paper_id=paper.id,
                    section=sec_name,
                    text=chunk_text,
                    char_start=start_pos,
                    char_end=end_pos,
                    token_count=len(chunk_text.split()),
                    metadata={
                        "paper_id": paper.id,
                        "title": paper.title,
                        "year": paper.year,
                        "fields_of_study": paper.fields_of_study,
                    },
                )
            )

        return chunks
