from __future__ import annotations

import hashlib
import logging
import math
from typing import Sequence
from contracts.schemas.models import Chunk
from ai.config import settings

logger = logging.getLogger("aurelium.ai.ingestion.embedder")


def _deterministic_pseudo_vector(text: str, dim: int) -> list[float]:
    """Generates an L2-normalized deterministic vector for offline/testing environments."""
    values = []
    for i in range(dim):
        seed = f"{text}:{i}"
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        val = (int(digest[:8], 16) / 0xFFFFFFFF) * 2.0 - 1.0
        values.append(val)

    # L2 normalize
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [round(v / norm, 6) for v in values]


class ChunkEmbedder:
    """
    Embeds Chunk instances using a sentence-transformers model.
    Ensures vectors are L2-normalized, matching EMBEDDING_DIM.
    Processes in batches and skips chunks already stored.
    """

    def __init__(
        self,
        model_name: str | None = None,
        embedding_dim: int | None = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name or settings.embedding_model
        self.embedding_dim = embedding_dim or settings.embedding_dim
        self.batch_size = batch_size
        self._model = None

    def _get_model(self):
        """Lazily loads sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                logger.info(f"Loading local embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:
                logger.warning(
                    f"Could not load sentence-transformers model '{self.model_name}' ({exc}); "
                    "using fallback deterministic embedding generator."
                )
                self._model = False
        return self._model

    def filter_unstored_chunks(
        self,
        chunks: Sequence[Chunk],
        stored_chunk_ids: set[str],
    ) -> list[Chunk]:
        """Filters out chunks that have already been embedded and stored."""
        unstored = [c for c in chunks if c.chunk_id not in stored_chunk_ids]
        skipped = len(chunks) - len(unstored)
        if skipped > 0:
            logger.info(f"Skipping {skipped} already-stored chunks from embedding batch.")
        return unstored

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Embeds a list of chunks in batches with L2 normalization.
        Mutates or returns copies of chunks with .embedding populated.
        """
        if not chunks:
            return []

        model = self._get_model()
        embedded_chunks: list[Chunk] = []

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            texts = [c.text for c in batch]

            if model:
                # Real sentence-transformers encoding with L2 normalization
                raw_vectors = model.encode(texts, normalize_embeddings=True)
                vectors = [v.tolist() for v in raw_vectors]
            else:
                # Deterministic fallback vectors
                vectors = [_deterministic_pseudo_vector(t, self.embedding_dim) for t in texts]

            for chunk, vec in zip(batch, vectors):
                # Ensure dimension matches target EMBEDDING_DIM
                if len(vec) != self.embedding_dim:
                    # Pad or slice if model dim differs from config dim
                    if len(vec) < self.embedding_dim:
                        vec = vec + [0.0] * (self.embedding_dim - len(vec))
                    else:
                        vec = vec[: self.embedding_dim]
                    # Re-normalize
                    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
                    vec = [round(v / norm, 6) for v in vec]

                embedded_chunks.append(chunk.model_copy(update={"embedding": vec}))

        logger.info(f"Successfully embedded {len(embedded_chunks)} chunks.")
        return embedded_chunks
