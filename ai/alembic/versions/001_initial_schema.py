"""001_initial_schema

Create papers, chunks (with TSVECTOR GIN index), and research_runs tables.
Vectors live exclusively in Pinecone (no pgvector extension used).

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-10-10 22:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Papers table
    op.create_table(
        "papers",
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("abstract", sa.Text(), nullable=False),
        sa.Column("authors", postgresql.JSONB(astext_type=sa.Text()), server_default="[]", nullable=False),
        sa.Column("venue", sa.String(), nullable=True),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("published_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("doi", sa.String(), nullable=True),
        sa.Column("arxiv_id", sa.String(), nullable=True),
        sa.Column("semantic_scholar_id", sa.String(), nullable=True),
        sa.Column("citation_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("fields_of_study", postgresql.JSONB(astext_type=sa.Text()), server_default="[]", nullable=False),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), server_default="{}", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_papers_id", "papers", ["id"], unique=False)
    op.create_index("ix_papers_year", "papers", ["year"], unique=False)
    op.create_index("ix_papers_published_date", "papers", ["published_date"], unique=False)
    op.create_index("ix_papers_doi", "papers", ["doi"], unique=True)
    op.create_index("ix_papers_arxiv_id", "papers", ["arxiv_id"], unique=True)
    op.create_index("ix_papers_semantic_scholar_id", "papers", ["semantic_scholar_id"], unique=True)

    # 2. Chunks table (with TSVECTOR column and GIN index for keyword / BM25 search)
    op.create_table(
        "chunks",
        sa.Column("chunk_id", sa.String(), primary_key=True, nullable=False),
        sa.Column("paper_id", sa.String(), sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("section", sa.String(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), server_default="{}", nullable=False),
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', text)", persisted=True),
            nullable=True,
        ),
    )
    op.create_index("ix_chunks_chunk_id", "chunks", ["chunk_id"], unique=False)
    op.create_index("ix_chunks_paper_id", "chunks", ["paper_id"], unique=False)
    op.create_index("idx_chunks_tsv", "chunks", ["tsv"], postgresql_using="gin")

    # 3. Research Runs table
    op.create_table(
        "research_runs",
        sa.Column("run_id", sa.String(), primary_key=True, nullable=False),
        sa.Column("status", sa.String(), server_default="pending", nullable=False),
        sa.Column("state", postgresql.JSONB(astext_type=sa.Text()), server_default="{}", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_research_runs_run_id", "research_runs", ["run_id"], unique=False)
    op.create_index("ix_research_runs_status", "research_runs", ["status"], unique=False)


def downgrade() -> None:
    op.drop_table("research_runs")
    op.drop_table("chunks")
    op.drop_table("papers")
