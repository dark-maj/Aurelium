from __future__ import annotations

from typing import List, Optional, Tuple, Any
from schemas.models import Paper, Chunk

# SQL schema definitions for Postgres + pgvector
PGVECTOR_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    authors TEXT[] DEFAULT '{}',
    venue TEXT,
    year INT,
    published_date TIMESTAMP WITH TIME ZONE,
    doi TEXT,
    arxiv_id TEXT,
    fields_of_study TEXT[] DEFAULT '{}',
    citation_count INT DEFAULT 0,
    url TEXT,
    embedding vector(768),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    paper_id TEXT REFERENCES papers(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    section TEXT,
    token_count INT,
    embedding vector(768),
    metadata JSONB DEFAULT '{}'::jsonb,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

CREATE INDEX IF NOT EXISTS idx_papers_embedding ON papers USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING gin(tsv);
"""


class InMemoryVectorStore:
    """
    Lightweight in-memory vector store for prototyping, development, and testing
    when a live Postgres + pgvector instance is not connected.
    """

    def __init__(self):
        self.papers: dict[str, Paper] = {}
        self.chunks: dict[str, Chunk] = {}

    def upsert_paper(self, paper: Paper) -> None:
        self.papers[paper.id] = paper

    def upsert_chunks(self, chunks: List[Chunk]) -> None:
        for chunk in chunks:
            self.chunks[chunk.id] = chunk

    def search_candidates(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 20,
    ) -> List[Tuple[Paper, float]]:
        """
        Retrieves top candidate papers. In production with pgvector, executes:
        SELECT p.*, 1 - (p.embedding <=> :query_vec) AS score FROM papers p ORDER BY score DESC LIMIT :k
        """
        results: List[Tuple[Paper, float]] = []
        for paper in self.papers.values():
            # Simple keyword matching fallback
            query_words = set(query.lower().split())
            text = f"{paper.title} {paper.abstract}".lower()
            matches = sum(1 for w in query_words if w in text)
            score = round(matches / max(1, len(query_words)), 4)
            results.append((paper, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
