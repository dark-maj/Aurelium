from __future__ import annotations

import re
import unicodedata
from contracts.schemas.models import Paper


def clean_text(text: str | None) -> str:
    """
    Cleans academic text by removing HTML tags, LaTeX macros, and odd unicode.
    Strictly preserves original casing (required for SciBERT/SPECTER case sensitivity).
    """
    if not text:
        return ""

    # 1. Unicode normalization (NFKC)
    cleaned = unicodedata.normalize("NFKC", text)

    # 2. Strip HTML tags
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)

    # 3. Strip LaTeX math blocks
    cleaned = re.sub(r"\$\$.*?\$\$", " ", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\$.*?\$", " ", cleaned)

    # 4. Strip common formatting commands preserving inner content: \command{content} -> content
    cleaned = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", cleaned)

    # 5. Remove remaining LaTeX macros: \ref, \label, \vspace, etc.
    cleaned = re.sub(r"\\[a-zA-Z]+", " ", cleaned)

    # 6. Unescape common LaTeX symbols
    cleaned = re.sub(r"\\([&%$#_{}])", r"\1", cleaned)

    # 7. Remove non-printable / control characters (except standard whitespace)
    cleaned = "".join(ch for ch in cleaned if ch == "\n" or ch == "\t" or not unicodedata.category(ch).startswith("C"))

    # 8. Collapse excessive whitespace
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n", "\n\n", cleaned)

    return cleaned.strip()


class PaperCleaner:
    """Validates and cleans ingested papers according to Aurelium quality filters."""

    MIN_ABSTRACT_WORDS = 50

    @classmethod
    def clean_paper(cls, paper: Paper) -> Paper | None:
        """
        Cleans title and abstract while keeping original casing.
        Drops papers with missing abstracts or abstracts under 50 words.
        Returns cleaned Paper or None if dropped.
        """
        if not paper.abstract or not paper.abstract.strip():
            return None

        cleaned_abstract = clean_text(paper.abstract)
        words = cleaned_abstract.split()
        if len(words) < cls.MIN_ABSTRACT_WORDS:
            return None

        cleaned_title = clean_text(paper.title)

        # Return updated copy with cleaned content and intact casing/dates
        return paper.model_copy(
            update={
                "title": cleaned_title,
                "abstract": cleaned_abstract,
            }
        )

    @classmethod
    def clean_papers(cls, papers: list[Paper]) -> list[Paper]:
        """Filters and cleans a batch of papers."""
        cleaned_list: list[Paper] = []
        for p in papers:
            cleaned = cls.clean_paper(p)
            if cleaned is not None:
                cleaned_list.append(cleaned)
        return cleaned_list
