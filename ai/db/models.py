from __future__ import annotations

from datetime import datetime
from typing import Any
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Computed,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class PaperModel(Base):
    __tablename__ = "papers"

    id = Column(String, primary_key=True, index=True)
    title = Column(Text, nullable=False)
    abstract = Column(Text, nullable=False)
    authors = Column(JSONB, default=list, nullable=False)
    venue = Column(String, nullable=True)
    year = Column(Integer, nullable=True, index=True)
    published_date = Column(DateTime(timezone=True), nullable=True, index=True)
    doi = Column(String, nullable=True, unique=True, index=True)
    arxiv_id = Column(String, nullable=True, unique=True, index=True)
    semantic_scholar_id = Column(String, nullable=True, unique=True, index=True)
    citation_count = Column(Integer, default=0, nullable=False)
    fields_of_study = Column(JSONB, default=list, nullable=False)
    url = Column(Text, nullable=True)
    metadata_json = Column("metadata", JSONB, default=dict, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    chunks = relationship("ChunkModel", back_populates="paper", cascade="all, delete-orphan")


class ChunkModel(Base):
    __tablename__ = "chunks"

    chunk_id = Column(String, primary_key=True, index=True)
    paper_id = Column(String, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True)
    section = Column(String, nullable=True)
    text = Column(Text, nullable=False)
    char_start = Column(Integer, nullable=False)
    char_end = Column(Integer, nullable=False)
    token_count = Column(Integer, nullable=True)
    metadata_json = Column("metadata", JSONB, default=dict, nullable=False)

    # Generated TSVECTOR column for full-text search with English stemming
    tsv = Column(
        TSVECTOR,
        Computed("to_tsvector('english', text)", persisted=True),
        nullable=True,
    )

    paper = relationship("PaperModel", back_populates="chunks")

    __table_args__ = (
        Index("idx_chunks_tsv", "tsv", postgresql_using="gin"),
    )


class ResearchRunModel(Base):
    __tablename__ = "research_runs"

    run_id = Column(String, primary_key=True, index=True)
    status = Column(String, nullable=False, default="pending", index=True)
    state = Column(JSONB, default=dict, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
