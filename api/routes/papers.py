from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from schemas.models import Paper, Chunk
from ingestion.cleaner import TextCleaner
from ingestion.chunker import Chunker
from ingestion.deduplication import Deduplicator
from retrieval.vector_store import InMemoryVectorStore
from api.deps import get_vector_store

router = APIRouter(prefix="/papers", tags=["Papers & Ingestion"])
deduplicator = Deduplicator()
chunker = Chunker()


class IngestPaperRequest(BaseModel):
    paper: Paper


class IngestBatchResponse(BaseModel):
    ingested_count: int
    chunks_created: int
    paper_ids: List[str]


@router.post("/", response_model=Paper, status_code=status.HTTP_201_CREATED)
async def ingest_single_paper(
    payload: IngestPaperRequest,
    store: InMemoryVectorStore = Depends(get_vector_store),
):
    paper = payload.paper
    if deduplicator.is_duplicate(paper):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Paper with DOI '{paper.doi}' or title '{paper.title}' already exists.",
        )

    deduplicator.register(paper)
    store.upsert_paper(paper)

    # Generate and store chunks
    chunks = chunker.chunk_paper(paper)
    store.upsert_chunks(chunks)

    return paper


@router.post("/batch", response_model=IngestBatchResponse)
async def ingest_batch_papers(
    papers: List[Paper],
    store: InMemoryVectorStore = Depends(get_vector_store),
):
    unique_papers = deduplicator.filter_unique(papers)
    all_chunks: List[Chunk] = []

    for paper in unique_papers:
        store.upsert_paper(paper)
        chunks = chunker.chunk_paper(paper)
        all_chunks.extend(chunks)

    store.upsert_chunks(all_chunks)

    return IngestBatchResponse(
        ingested_count=len(unique_papers),
        chunks_created=len(all_chunks),
        paper_ids=[p.id for p in unique_papers],
    )
