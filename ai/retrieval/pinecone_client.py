from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional, Sequence
from pydantic import BaseModel, Field

from ai.config import settings

logger = logging.getLogger("aurelium.ai.pinecone")


class PineconeMatch(BaseModel):
    """Normalized match object returned from vector queries."""
    id: str
    score: float
    paper_id: str
    chunk_id: str
    year: Optional[int] = None
    fields_of_study: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorRecord(BaseModel):
    """Input representation of a vector to upsert."""
    id: str
    values: list[float]
    paper_id: str
    chunk_id: str
    year: Optional[int] = None
    fields_of_study: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


try:
    from pinecone import Pinecone, ServerlessSpec  # type: ignore
except ImportError:
    Pinecone = None  # type: ignore

    class ServerlessSpec:  # type: ignore
        def __init__(self, cloud: str, region: str):
            self.cloud = cloud
            self.region = region


def ensure_pinecone_index(
    client_instance: Any,
    index_name: str | None = None,
    dimension: int | None = None,
    metric: str = "cosine",
    cloud: str | None = None,
    region: str | None = None,
) -> None:
    """
    Checks whether the target index exists in Pinecone; creates it if missing
    using serverless specs.
    """
    idx_name = index_name or settings.pinecone_index
    dim = dimension or settings.embedding_dim
    c_cloud = cloud or settings.pinecone_cloud
    c_region = region or settings.pinecone_region

    try:
        existing_indices = client_instance.list_indexes()
        # Handle both list of strings or list of index models
        names = [
            idx.name if hasattr(idx, "name") else str(idx)
            for idx in (existing_indices.indexes if hasattr(existing_indices, "indexes") else existing_indices)
        ]

        if idx_name not in names:
            logger.info("Creating missing Pinecone index", extra={"index": idx_name, "dim": dim})
            client_instance.create_index(
                name=idx_name,
                dimension=dim,
                metric=metric,
                spec=ServerlessSpec(cloud=c_cloud, region=c_region),
            )
            logger.info("Successfully initiated Pinecone index creation", extra={"index": idx_name})
        else:
            logger.debug("Pinecone index already exists", extra={"index": idx_name})
    except Exception as exc:
        logger.error("Error verifying or creating Pinecone index", extra={"error": str(exc)})
        raise


class PineconeVectorClient:
    """
    Async-friendly Pinecone wrapper.
    Executes blocking SDK operations within asyncio.to_thread to preserve event loop concurrency.
    Always injects required metadata: paper_id, chunk_id, year, and fields_of_study.
    """

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str | None = None,
        default_namespace: str | None = None,
        client_instance: Any = None,
    ):
        self.api_key = api_key or settings.pinecone_api_key
        self.index_name = index_name or settings.pinecone_index
        self.default_namespace = default_namespace or settings.pinecone_namespace

        if client_instance is not None:
            self._client = client_instance
            self._index = self._client.Index(self.index_name)
        else:
            try:
                from pinecone import Pinecone  # type: ignore
                self._client = Pinecone(api_key=self.api_key)
                self._index = self._client.Index(self.index_name)
            except (ImportError, Exception):
                self._client = None
                self._index = None

    async def ensure_index_ready(self) -> None:
        """Asynchronously runs ensure_pinecone_index."""
        await asyncio.to_thread(ensure_pinecone_index, self._client, self.index_name, settings.embedding_dim)

    async def upsert_records(
        self,
        records: Sequence[VectorRecord],
        namespace: str | None = None,
    ) -> int:
        """
        Upserts vector records to Pinecone.
        Guarantees that paper_id, chunk_id, year, and fields_of_study are set in metadata.
        """
        ns = namespace or self.default_namespace
        formatted_vectors: list[dict[str, Any]] = []

        for rec in records:
            meta = dict(rec.metadata)
            meta["paper_id"] = str(rec.paper_id)
            meta["chunk_id"] = str(rec.chunk_id)
            meta["year"] = int(rec.year) if rec.year is not None else 0
            meta["fields_of_study"] = list(rec.fields_of_study)

            formatted_vectors.append({
                "id": rec.id,
                "values": rec.values,
                "metadata": meta,
            })

        def _do_upsert() -> int:
            resp = self._index.upsert(vectors=formatted_vectors, namespace=ns)
            upserted = resp.get("upserted_count", len(formatted_vectors)) if isinstance(resp, dict) else len(formatted_vectors)
            return upserted

        count = await asyncio.to_thread(_do_upsert)
        logger.info("Upserted vectors to Pinecone", extra={"count": count, "namespace": ns})
        return count

    async def query(
        self,
        vector: list[float],
        top_k: int = 10,
        namespace: str | None = None,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[PineconeMatch]:
        """
        Queries Pinecone index within a specific namespace.
        """
        ns = namespace or self.default_namespace

        def _do_query() -> dict[str, Any]:
            return self._index.query(
                vector=vector,
                top_k=top_k,
                namespace=ns,
                filter=filter_dict,
                include_metadata=True,
            )

        resp = await asyncio.to_thread(_do_query)
        matches: list[PineconeMatch] = []

        raw_matches = resp.get("matches", []) if isinstance(resp, dict) else getattr(resp, "matches", [])
        for m in raw_matches:
            m_id = m.get("id") if isinstance(m, dict) else m.id
            m_score = float(m.get("score", 0.0) if isinstance(m, dict) else m.score)
            m_meta = dict(m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {}))

            matches.append(
                PineconeMatch(
                    id=m_id,
                    score=m_score,
                    paper_id=m_meta.get("paper_id", ""),
                    chunk_id=m_meta.get("chunk_id", ""),
                    year=m_meta.get("year"),
                    fields_of_study=m_meta.get("fields_of_study", []),
                    metadata=m_meta,
                )
            )

        logger.debug("Executed Pinecone query", extra={"returned": len(matches), "namespace": ns})
        return matches

    async def delete(
        self,
        ids: list[str] | None = None,
        delete_all: bool = False,
        namespace: str | None = None,
    ) -> None:
        """Deletes vectors by IDs or drops entire namespace."""
        ns = namespace or self.default_namespace

        def _do_delete() -> None:
            if delete_all:
                self._index.delete(delete_all=True, namespace=ns)
            elif ids:
                self._index.delete(ids=ids, namespace=ns)

        await asyncio.to_thread(_do_delete)
        logger.info("Deleted vectors from Pinecone", extra={"ids_count": len(ids) if ids else 0, "all": delete_all, "namespace": ns})


def get_pinecone_client() -> PineconeVectorClient:
    """Factory dependency for PineconeVectorClient."""
    return PineconeVectorClient()
