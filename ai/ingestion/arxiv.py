from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any
import httpx

from contracts.schemas.models import Paper
from ai.ingestion.rate_limiter import AsyncRateLimiter, execute_with_backoff

logger = logging.getLogger("aurelium.ai.ingestion.arxiv")

ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def parse_arxiv_id(raw_id_url: str) -> str:
    """Extracts normalized arXiv ID without version, e.g. '2305.18290v1' -> '2305.18290'."""
    match = re.search(r"(\d{4}\.\d{4,5}|[a-zA-Z\-]+(?:\.[a-zA-Z]+)?/\d{7})(?:v\d+)?", raw_id_url)
    if match:
        return match.group(1)
    return raw_id_url.rsplit("/", 1)[-1].split("v")[0]


def parse_arxiv_entry(entry: ET.Element) -> Paper:
    """Parses an Atom <entry> element into a contracts Paper instance."""
    id_elem = entry.find("atom:id", ATOM_NS)
    raw_id = id_elem.text.strip() if id_elem is not None and id_elem.text else ""
    arxiv_id = parse_arxiv_id(raw_id)

    title_elem = entry.find("atom:title", ATOM_NS)
    title = re.sub(r"\s+", " ", title_elem.text.strip()) if title_elem is not None and title_elem.text else ""

    summary_elem = entry.find("atom:summary", ATOM_NS)
    abstract = summary_elem.text.strip() if summary_elem is not None and summary_elem.text else ""

    # Authors
    authors = []
    for author_elem in entry.findall("atom:author", ATOM_NS):
        name_elem = author_elem.find("atom:name", ATOM_NS)
        if name_elem is not None and name_elem.text:
            authors.append(name_elem.text.strip())

    # Published date
    published_elem = entry.find("atom:published", ATOM_NS)
    published_date = None
    year = None
    if published_elem is not None and published_elem.text:
        try:
            # Format: '2023-05-29T17:59:00Z'
            published_date = datetime.fromisoformat(published_elem.text.strip().replace("Z", "+00:00"))
            year = published_date.year
        except Exception:
            pass

    # External DOI and Journal Ref
    doi_elem = entry.find("arxiv:doi", ATOM_NS)
    doi = doi_elem.text.strip() if doi_elem is not None and doi_elem.text else None

    journal_elem = entry.find("arxiv:journal_ref", ATOM_NS)
    venue = journal_elem.text.strip() if journal_elem is not None and journal_elem.text else "arXiv"

    # Categories / Fields of Study
    fields_of_study = []
    for cat in entry.findall("atom:category", ATOM_NS):
        term = cat.get("term")
        if term:
            fields_of_study.append(term)

    # Links: landing page and PDF
    pdf_url = None
    url = raw_id
    for link in entry.findall("atom:link", ATOM_NS):
        href = link.get("href")
        rel = link.get("rel")
        title_attr = link.get("title")
        if title_attr == "pdf" or (href and href.endswith(".pdf")):
            pdf_url = href
        elif rel == "alternate" and href:
            url = href

    metadata: dict[str, Any] = {
        "source": "arxiv",
        "raw_id": raw_id,
        "pdf_url": pdf_url,
        "references": [],
    }

    return Paper(
        id=f"arxiv:{arxiv_id}",
        title=title,
        abstract=abstract,
        authors=authors,
        venue=venue,
        year=year,
        published_date=published_date,
        doi=doi,
        arxiv_id=arxiv_id,
        semantic_scholar_id=None,
        citation_count=0,
        fields_of_study=fields_of_study,
        url=url,
        metadata=metadata,
    )


class ArxivClient:
    """Async client for fetching research papers from the arXiv API."""

    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(
        self,
        requests_per_second: float = 0.33,
        client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ):
        self.rate_limiter = AsyncRateLimiter(requests_per_second)
        self.timeout = timeout
        self._external_client = client

    async def fetch_papers(self, topic: str, limit: int = 50) -> list[Paper]:
        """
        Fetches papers matching the topic query, respecting rate limits and exponential backoff.
        """
        params = {
            "search_query": f"all:{topic}",
            "start": 0,
            "max_results": min(limit, 300),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        async def _do_request(c: httpx.AsyncClient) -> httpx.Response:
            await self.rate_limiter.acquire()
            return await c.get(self.BASE_URL, params=params, timeout=self.timeout)

        if self._external_client is not None:
            resp = await execute_with_backoff(lambda: _do_request(self._external_client), service_name="arXiv")
            xml_text = resp.text
        else:
            async with httpx.AsyncClient() as client:
                resp = await execute_with_backoff(lambda: _do_request(client), service_name="arXiv")
                xml_text = resp.text

        papers: list[Paper] = []
        try:
            root = ET.fromstring(xml_text)
            for entry in root.findall("atom:entry", ATOM_NS):
                paper = parse_arxiv_entry(entry)
                papers.append(paper)
        except Exception as exc:
            logger.error("Failed to parse arXiv Atom XML", extra={"error": str(exc)})
            raise

        logger.info(f"Fetched {len(papers)} papers from arXiv for topic '{topic}'")
        return papers
