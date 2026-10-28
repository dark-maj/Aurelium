import re
import unicodedata
from typing import Optional


class TextCleaner:
    """Cleans academic text by removing LaTeX artifacts, HTML, and normalizing unicode."""

    @staticmethod
    def clean(text: Optional[str]) -> str:
        if not text:
            return ""

        # Normalize unicode (NFKC)
        cleaned = unicodedata.normalize("NFKC", text)

        # Remove HTML tags
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)

        # Remove inline math ($...$) and block math ($$...$$)
        cleaned = re.sub(r"\$\$.*?\$\$", " [MATH] ", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"\$.*?\$", " [MATH] ", cleaned)

        # Remove common LaTeX commands (e.g. \textbf{...}, \cite{...})
        cleaned = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", cleaned)
        cleaned = re.sub(r"\\[a-zA-Z]+", " ", cleaned)

        # Collapse excessive whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

    @staticmethod
    def format_specter_input(title: str, abstract: str) -> str:
        """Formats title and abstract with [SEP] token as expected by SPECTER/SciBERT."""
        cleaned_title = TextCleaner.clean(title)
        cleaned_abstract = TextCleaner.clean(abstract)
        return f"{cleaned_title} [SEP] {cleaned_abstract}"
