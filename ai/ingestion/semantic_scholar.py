from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
import httpx

from contracts.schemas.models import Paper
from ai.ingestion.rate_limiter import AsyncRateLimiter, execute_with_backoff

logger = logging.getLogger("aurelium.ai.ingestion.semantic_scholar")


def parse_s2_paper(item: dict[str, Any]) -> Paper:
    """Parses a Semantic Scholar paper JSON object into a contracts Paper instance."""
    paper_id = item.get("paperId", "")
    title = (item.get("title") or "").strip()
    abstract = (item.get("abstract") or "").strip()

    authors = [
        author.get("name", "").strip()
        for author in item.get("authors", [])
        if author.get("name")
    ]

    year = item.get("year")
    pub_date_str = item.get("publicationDate")
    published_date = None
    if pub_date_str:
        try:
            # Handles 'YYYY-MM-DD'
            dt = datetime.strptime(pub_date_str, "%Y-%m-%d")
            published_date = dt.replace(tzinfo=timezone.utc)
            if not year:
                year = published_date.year
        except Exception:
            pass

    external_ids = item.get("externalIds") or {}
    doi = external_ids.get("DOI")
    arxiv_id = external_ids.get("ArXiv")
    corpus_id = str(external_ids.get("CorpusId")) if external_ids.get("CorpusId") else None

    venue = item.get("venue") or None
    citation_count = int(item.get("citationCount") or 0)
    fields_of_study = item.get("fieldsOfStudy") or []
    url = item.get("url")

    # Open Access PDF link if available
    pdf_info = item.get("openAccessPdf") or {}
    pdf_url = pdf_info.get("url")

    # Canonical ID: prefer DOI or arXiv ID if present, otherwise s2:{paper_id}
    canonical_id = f"doi:{doi}" if doi else (f"arxiv:{arxiv_id}" if arxiv_id else f"s2:{paper_id}")

    metadata: dict[str, Any] = {
        "source": "semanticscholar",
        "semantic_scholar_id": paper_id,
        "corpus_id": corpus_id,
        "reference_count": item.get("referenceCount", 0),
        "pdf_url": pdf_url,
    }

    return Paper(
        id=canonical_id,
        title=title,
        abstract=abstract,
        authors=authors,
        venue=venue,
        year=year,
        published_date=published_date,
        doi=doi,
        arxiv_id=arxiv_id,
        semantic_scholar_id=paper_id,
        citation_count=citation_count,
        fields_of_study=fields_of_study,
        url=url,
        metadata=metadata,
    )


class SemanticScholarClient:
    """Async client for fetching research papers from the Semantic Scholar Graph API."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    FIELDS = (
        "paperId,title,abstract,authors,year,venue,citationCount,referenceCount,"
        "fieldsOfStudy,publicationDate,externalIds,url,openAccessPdf"
    )

    def __init__(
        self,
        requests_per_second: float = 1.0,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ):
        self.rate_limiter = AsyncRateLimiter(requests_per_second)
        self.api_key = api_key
        self.timeout = timeout
        self._external_client = client

    async def fetch_papers(self, topic: str, limit: int = 50) -> list[Paper]:
        """Fetches papers matching the search topic with rate limiting and backoff."""
        params = {
            "query": topic,
            "limit": min(limit, 100),
            "fields": self.FIELDS,
        }
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        async def _do_request(c: httpx.AsyncClient) -> httpx.Response:
            await self.rate_limiter.acquire()
            return await c.get(self.BASE_URL, params=params, headers=headers, timeout=self.timeout)

        if self._external_client is not None:
            resp = await execute_with_backoff(
                lambda: _do_request(self._external_client), service_name="SemanticScholar"
            )
            data = resp.json()
        else:
            async with httpx.AsyncClient() as client:
                resp = await execute_with_backoff(
                    lambda: _do_request(client), service_name="SemanticScholar"
                )
                data = resp.json()

        raw_papers = data.get("data", [])
        papers: list[Paper] = [parse_s2_paper(item) for item in raw_papers]
        logger.info(f"Fetched {len(papers)} papers from Semantic Scholar for topic '{topic}'")
        return papers
