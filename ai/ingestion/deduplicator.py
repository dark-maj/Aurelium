from __future__ import annotations

import difflib
import logging
import re
from contracts.schemas.models import Paper

logger = logging.getLogger("aurelium.ai.ingestion.deduplicator")


def normalize_doi(doi: str | None) -> str | None:
    """Normalizes DOI by stripping URI prefix, trailing slashes, and casing."""
    if not doi:
        return None
    cleaned = doi.strip().lower()
    cleaned = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", cleaned)
    return cleaned.strip()


def normalize_arxiv_id(arxiv_id: str | None) -> str | None:
    """Normalizes arXiv ID by removing version suffixes and URL/prefix elements."""
    if not arxiv_id:
        return None
    cleaned = arxiv_id.strip().lower()
    cleaned = re.sub(r"^arxiv:", "", cleaned)
    # Strip trailing version (e.g., 2305.18290v2 -> 2305.18290)
    cleaned = re.sub(r"v\d+$", "", cleaned)
    return cleaned.strip()


def normalize_title(title: str | None) -> str:
    """Normalizes title for fuzzy comparison: lowercase, alphanumeric tokens only."""
    if not title:
        return ""
    cleaned = re.sub(r"[^\w\s]", "", title.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def merge_papers(target: Paper, source: Paper) -> Paper:
    """
    Merges information from source into target, preserving the richest attributes
    (external IDs, publication date, citation counts, PDF URLs).
    """
    updated_doi = target.doi or source.doi
    updated_arxiv = target.arxiv_id or source.arxiv_id
    updated_s2 = target.semantic_scholar_id or source.semantic_scholar_id

    # Preserve highest citation count
    updated_citations = max(target.citation_count, source.citation_count)

    # Preserve earliest or non-null publication date
    updated_pub_date = target.published_date or source.published_date
    updated_year = target.year or source.year or (updated_pub_date.year if updated_pub_date else None)

    # Merge author lists (keep longer list)
    updated_authors = target.authors if len(target.authors) >= len(source.authors) else source.authors

    # Merge fields of study
    merged_fields = list(dict.fromkeys(target.fields_of_study + source.fields_of_study))

    # Merge metadata
    merged_metadata = dict(target.metadata)
    for k, v in source.metadata.items():
        if k not in merged_metadata or not merged_metadata[k]:
            merged_metadata[k] = v

    return target.model_copy(
        update={
            "doi": updated_doi,
            "arxiv_id": updated_arxiv,
            "semantic_scholar_id": updated_s2,
            "citation_count": updated_citations,
            "published_date": updated_pub_date,
            "year": updated_year,
            "authors": updated_authors,
            "fields_of_study": merged_fields,
            "metadata": merged_metadata,
        }
    )


class Deduplicator:
    """
    3-stage deduplicator across academic sources:
    1. DOI exact match
    2. arXiv ID exact match (version-normalized)
    3. Normalized title fuzzy similarity match (ratio >= threshold)
    """

    def __init__(self, fuzzy_threshold: float = 0.88):
        self.fuzzy_threshold = fuzzy_threshold
        self.doi_map: dict[str, str] = {}         # norm_doi -> paper_id
        self.arxiv_map: dict[str, str] = {}       # norm_arxiv_id -> paper_id
        self.title_map: dict[str, str] = {}       # norm_title -> paper_id
        self.papers: dict[str, Paper] = {}        # paper_id -> Paper

    def register(self, paper: Paper) -> None:
        """Indexes paper identifiers for subsequent duplicate checks."""
        self.papers[paper.id] = paper

        norm_doi = normalize_doi(paper.doi)
        if norm_doi:
            self.doi_map[norm_doi] = paper.id

        norm_arxiv = normalize_arxiv_id(paper.arxiv_id)
        if norm_arxiv:
            self.arxiv_map[norm_arxiv] = paper.id

        norm_t = normalize_title(paper.title)
        if norm_t:
            self.title_map[norm_t] = paper.id

    def find_match(self, paper: Paper) -> Paper | None:
        """Finds existing matching Paper using the 3-stage protocol."""
        # Stage 1: DOI match
        norm_doi = normalize_doi(paper.doi)
        if norm_doi and norm_doi in self.doi_map:
            matched_id = self.doi_map[norm_doi]
            return self.papers[matched_id]

        # Stage 2: arXiv ID match
        norm_arxiv = normalize_arxiv_id(paper.arxiv_id)
        if norm_arxiv and norm_arxiv in self.arxiv_map:
            matched_id = self.arxiv_map[norm_arxiv]
            return self.papers[matched_id]

        # Stage 3: Fuzzy title match
        norm_t = normalize_title(paper.title)
        if norm_t:
            # Exact normalized title match first
            if norm_t in self.title_map:
                matched_id = self.title_map[norm_t]
                return self.papers[matched_id]

            # Fuzzy ratio comparison
            for existing_title, existing_id in self.title_map.items():
                # Fast pre-filter on length difference
                if abs(len(existing_title) - len(norm_t)) / max(len(existing_title), len(norm_t)) > 0.25:
                    continue
                ratio = difflib.SequenceMatcher(None, norm_t, existing_title).ratio()
                if ratio >= self.fuzzy_threshold:
                    logger.debug(
                        "Fuzzy title match detected",
                        extra={"ratio": round(ratio, 3), "t1": norm_t, "t2": existing_title},
                    )
                    return self.papers[existing_id]

        return None

    def deduplicate(self, papers: list[Paper]) -> list[Paper]:
        """
        Deduplicates incoming batch, merging duplicate metadata into canonical instances.
        Returns deduplicated unique papers list.
        """
        for incoming in papers:
            existing = self.find_match(incoming)
            if existing is not None:
                # Merge into existing paper
                merged = merge_papers(existing, incoming)
                self.papers[existing.id] = merged
                self.register(merged)
            else:
                self.register(incoming)

        return list(self.papers.values())
