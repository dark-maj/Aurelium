from __future__ import annotations

import logging
from typing import Sequence

from contracts.interfaces.protocols import RankerProtocol
from contracts.interfaces.stubs import StubRanker
from contracts.schemas.models import Paper
from ai.retrieval.models import ScoredChunk

logger = logging.getLogger("aurelium.ai.retrieval.reranker")


class CrossEncoderReranker:
    """Optional cross-encoder reranker, off by default, toggled via config."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore
                self._model = CrossEncoder(self.model_name)
            except Exception as exc:
                logger.debug(f"CrossEncoder unavailable ({exc}); using fallback scoring.")
                self._model = False
        return self._model

    def rerank(self, query: str, chunks: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        """Scores (query, chunk.text) pairs and reorders descending."""
        if not chunks:
            return []

        model = self._get_model()
        pairs = [(query, c.text) for c in chunks]

        if model:
            scores = model.predict(pairs)
        else:
            # Fallback: maintain relative RRF score with subtle keyword boost
            q_words = set(query.lower().split())
            scores = []
            for c in chunks:
                words = set(c.text.lower().split())
                overlap = len(q_words & words) / max(1, len(q_words))
                scores.append(round(c.final_score * 0.7 + overlap * 0.3, 4))

        reranked: list[ScoredChunk] = []
        for c, score in zip(chunks, scores):
            c_copy = c.model_copy(deep=True)
            c_copy.scores.cross_encoder_score = float(score)
            c_copy.final_score = float(score)
            reranked.append(c_copy)

        reranked.sort(key=lambda x: x.final_score, reverse=True)
        return reranked


def apply_ranker_protocol(
    query: str,
    chunks: Sequence[ScoredChunk],
    ranker: RankerProtocol | None = None,
) -> list[ScoredChunk]:
    """
    Hook passing the candidate papers through RankerProtocol from contracts
    and reordering by its scores. Defaults to StubRanker.
    """
    if not chunks:
        return []

    protocol_ranker = ranker or StubRanker()

    # Extract or synthesize Paper representations for the ranker protocol
    papers: list[Paper] = []
    for c in chunks:
        if c.paper is not None:
            papers.append(c.paper)
        else:
            papers.append(
                Paper(
                    id=c.paper_id,
                    title=f"Chunk {c.chunk_id}",
                    abstract=c.text,
                )
            )

    try:
        scores = protocol_ranker.rank(query=query, papers=papers)
    except Exception as exc:
        logger.error(f"Error executing RankerProtocol hook: {exc}")
        return list(chunks)

    reordered: list[ScoredChunk] = []
    for c, score in zip(chunks, scores):
        c_copy = c.model_copy(deep=True)
        c_copy.scores.ranker_score = float(score)
        c_copy.final_score = float(score)
        reordered.append(c_copy)

    # Reorder descending by ranker protocol score
    reordered.sort(key=lambda x: x.final_score, reverse=True)
    logger.debug(f"RankerProtocol reordered {len(reordered)} candidate chunks.")
    return reordered
