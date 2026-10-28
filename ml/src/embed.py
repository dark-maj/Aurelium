"""
Aurelium - Phase 2: SPECTER Embedding Pipeline
Generates 768-dimensional, L2-normalized paper representations from "title [SEP] abstract".
Caches vectors to disk and supports batching, GPU acceleration, and checkpoint resuming.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/allenai-specter"
DEFAULT_CACHE_PATH = "data/clean/embeddings.npy"
DEFAULT_IDS_CACHE_PATH = "data/clean/embedding_ids.json"


def prepare_text(title: Optional[str], abstract: Optional[str]) -> Tuple[str, bool]:
    """
    Format paper text for SPECTER.
    Returns (formatted_text, has_abstract_flag).
    If abstract is missing, embeds title only and marks has_abstract=False.
    """
    cleaned_title = (title or "").strip()
    cleaned_abstract = (abstract or "").strip()

    if cleaned_abstract:
        return f"{cleaned_title} [SEP] {cleaned_abstract}", True
    return cleaned_title, False


class SpecterEmbedder:
    """Manages SPECTER model loading, batch encoding, and disk caching."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading SPECTER model '{model_name}' on device '{self.device}'...")
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode list of strings into 768-dimensional L2-normalized vectors.
        Returns np.ndarray of shape (N, 768) with dtype float32.
        """
        if not texts:
            return np.empty((0, 768), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)


def embed_corpus(
    records: List[Dict[str, Any]],
    cache_path: str = DEFAULT_CACHE_PATH,
    ids_path: str = DEFAULT_IDS_CACHE_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
    device: Optional[str] = None,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Compute or load cached SPECTER embeddings for a list of paper records.

    Returns:
        embeddings: np.ndarray of shape (N, 768)
        ids: List[str] of paper IDs aligned with embeddings
        has_abstract_flags: np.ndarray of shape (N,) boolean
    """
    target_ids = [r.get("id") for r in records if r.get("id")]
    n_expected = len(target_ids)

    # 1. Check if complete cached embeddings exist
    if not force_recompute and os.path.exists(cache_path) and os.path.exists(ids_path):
        try:
            with open(ids_path, "r", encoding="utf-8") as f:
                cached_ids = json.load(f)

            if cached_ids == target_ids:
                print(f"Loading cached embeddings from {cache_path} ({len(cached_ids)} papers)...")
                cached_embs = np.load(cache_path)
                if cached_embs.shape[0] == n_expected:
                    has_abstract_flags = np.array(
                        [bool(r.get("abstract")) for r in records], dtype=bool
                    )
                    return cached_embs, target_ids, has_abstract_flags
        except Exception as e:
            print(f"Warning: Failed to load existing cache ({e}). Recomputing...")

    # 2. Prepare texts for embedding
    texts: List[str] = []
    has_abstract_list: List[bool] = []
    paper_ids: List[str] = []

    for r in records:
        pid = r.get("id")
        if not pid:
            continue
        text, has_abs = prepare_text(r.get("title"), r.get("abstract"))
        texts.append(text)
        has_abstract_list.append(has_abs)
        paper_ids.append(pid)

    has_abstract_flags = np.array(has_abstract_list, dtype=bool)

    if not texts:
        return np.empty((0, 768), dtype=np.float32), [], has_abstract_flags

    # 3. Compute embeddings
    embedder = SpecterEmbedder(model_name=model_name, device=device)
    print(f"Generating SPECTER embeddings for {len(texts)} papers (batch_size={batch_size})...")
    embeddings = embedder.embed_texts(texts, batch_size=batch_size, show_progress=True)

    # 4. Cache results to disk
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(ids_path)), exist_ok=True)

    np.save(cache_path, embeddings)
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(paper_ids, f)

    print(f"Cached {embeddings.shape} embeddings to {cache_path}")
    return embeddings, paper_ids, has_abstract_flags


def main():
    """Diagnostic CLI to run SPECTER embeddings on a sample dataset."""
    parser = argparse.ArgumentParser(description="Aurelium SPECTER Embedding Generator")
    parser.add_argument("--jsonl", default="data/test/raw/papers.jsonl", help="Input JSONL path")
    parser.add_argument("--out-npy", default="data/test/clean/embeddings.npy", help="Output .npy path")
    parser.add_argument("--out-ids", default="data/test/clean/embedding_ids.json", help="Output IDs JSON path")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for encoding")
    args = parser.parse_args()

    if not os.path.exists(args.jsonl):
        print(f"File not found: {args.jsonl}")
        return

    records = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records from {args.jsonl}")
    embs, ids, flags = embed_corpus(
        records,
        cache_path=args.out_npy,
        ids_path=args.out_ids,
        batch_size=args.batch_size,
    )
    print(f"Done! Embeddings shape: {embs.shape} | Has abstract: {flags.sum()}/{len(flags)}")


if __name__ == "__main__":
    main()
