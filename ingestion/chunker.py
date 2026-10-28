from typing import List, Optional
from uuid import uuid4
from schemas.models import Paper, Chunk
from ingestion.cleaner import TextCleaner


class Chunker:
    """Splits papers into retrievable chunks with metadata preservation."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_paper(self, paper: Paper) -> List[Chunk]:
        """
        Generates chunks for a paper. Creates primary abstract chunk,
        and sliding-window chunks for longer texts.
        """
        chunks: List[Chunk] = []
        cleaned_abstract = TextCleaner.clean(paper.abstract)

        # Primary abstract chunk
        abstract_chunk = Chunk(
            id=f"{paper.id}_chunk_0",
            paper_id=paper.id,
            chunk_index=0,
            content=f"Title: {paper.title}\n\nAbstract: {cleaned_abstract}",
            section="Abstract",
            token_count=len(cleaned_abstract.split()),
            metadata={
                "title": paper.title,
                "year": paper.year,
                "authors": paper.authors,
                "venue": paper.venue,
            }
        )
        chunks.append(abstract_chunk)

        # Additional paragraph/section chunking if body text is present in metadata
        body_text = paper.metadata.get("full_text")
        if body_text:
            cleaned_body = TextCleaner.clean(body_text)
            words = cleaned_body.split()
            step = max(1, self.chunk_size - self.overlap)
            idx = 1
            for start in range(0, len(words), step):
                chunk_words = words[start : start + self.chunk_size]
                if not chunk_words:
                    break
                chunk_content = " ".join(chunk_words)
                chunks.append(
                    Chunk(
                        id=f"{paper.id}_chunk_{idx}",
                        paper_id=paper.id,
                        chunk_index=idx,
                        content=chunk_content,
                        section=paper.metadata.get("section", "Body"),
                        token_count=len(chunk_words),
                        metadata={"title": paper.title, "year": paper.year}
                    )
                )
                idx += 1

        return chunks
