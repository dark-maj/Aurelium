import re
from typing import Dict, List, Optional
from schemas.models import Paper


class Deduplicator:
    """Deduplicates papers across arXiv, Semantic Scholar, and PubMed."""

    def __init__(self):
        self.seen_dois: Dict[str, str] = {}
        self.seen_titles: Dict[str, str] = {}

    @staticmethod
    def normalize_title(title: str) -> str:
        """Strip punctuation, casing, and spacing for fuzzy matching."""
        cleaned = re.sub(r"[^\w\s]", "", title.lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    def is_duplicate(self, paper: Paper) -> bool:
        if paper.doi:
            norm_doi = paper.doi.strip().lower()
            if norm_doi in self.seen_dois:
                return True

        norm_title = self.normalize_title(paper.title)
        if norm_title in self.seen_titles:
            return True

        return False

    def register(self, paper: Paper) -> None:
        if paper.doi:
            self.seen_dois[paper.doi.strip().lower()] = paper.id
        norm_title = self.normalize_title(paper.title)
        self.seen_titles[norm_title] = paper.id

    def filter_unique(self, papers: List[Paper]) -> List[Paper]:
        unique_papers: List[Paper] = []
        for p in papers:
            if not self.is_duplicate(p):
                self.register(p)
                unique_papers.append(p)
        return unique_papers
