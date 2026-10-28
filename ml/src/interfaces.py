"""
Aurelium - ML/AI Interface Protocols
Defines the contract between the Machine Learning models and the AI/Agentic layer.
"""

import json
import os
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ImpactProtocol(Protocol):
    """
    Protocol for citation impact prediction.
    Predicts the expected 3-year citation count for a research paper.
    """

    def predict(self, paper: Dict[str, Any]) -> float:
        """
        Given a paper record with metadata and text, return the predicted
        3-year citation count (raw count scale, NOT the log value).
        """
        ...


class StubImpact:
    """
    Stub implementation of ImpactProtocol returning a fixed dummy citation count.
    Enables the AI/RAG pipeline to run end-to-end before the full model is trained.
    """

    def __init__(self, default_score: float = 5.0):
        self.default_score = float(default_score)

    def predict(self, paper: Dict[str, Any]) -> float:
        """Return constant citation prediction."""
        return self.default_score


class ImpactModel:
    """
    Production implementation of ImpactProtocol.
    Loads trained Phase 2 model artifacts (LightGBM/MLP and preprocessors)
    and predicts citations for any given paper record.
    """

    def __init__(self, model_dir: str = "models/phase2"):
        self.model_dir = model_dir
        self.metadata_path = os.path.join(model_dir, "metadata.json")
        self.lgb_model_path = os.path.join(model_dir, "lightgbm_model.txt")
        self.preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")

        self.model = None
        self.preprocessor = None
        self.embedder = None
        self.metadata = {}

        self._load_artifacts()

    def _load_artifacts(self):
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(
                f"Phase 2 metadata not found at {self.metadata_path}. "
                "Ensure Phase 2 model training has been executed."
            )

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        import joblib
        import lightgbm as lgb

        if os.path.exists(self.preprocessor_path):
            self.preprocessor = joblib.load(self.preprocessor_path)

        if os.path.exists(self.lgb_model_path):
            self.model = lgb.Booster(model_file=self.lgb_model_path)
        else:
            raise FileNotFoundError(f"LightGBM model not found at {self.lgb_model_path}")

    def predict(self, paper: Dict[str, Any]) -> float:
        """
        Predict expected 3-year citation count for a single paper record.
        Extracts features, computes SPECTER embedding on-the-fly,
        runs the model, and reverses log1p to return the raw citation count.
        """
        from src.citation_model import extract_metadata_features
        from src.embed import SpecterEmbedder, prepare_text

        # 1. Compute text embedding
        if self.embedder is None:
            self.embedder = SpecterEmbedder()

        text, has_abs = prepare_text(paper.get("title"), paper.get("abstract"))
        emb = self.embedder.embed_texts([text], batch_size=1, show_progress=False)

        # 2. Extract metadata features
        meta_df = extract_metadata_features([paper])
        meta_feat = self.preprocessor.transform(meta_df)

        # 3. Concatenate features
        X = np.hstack([emb, meta_feat])

        # 4. Predict log1p citation count
        pred_log = float(self.model.predict(X)[0])

        # 5. Reverse log1p to raw non-negative citation count: exp(log_y) - 1
        raw_pred = max(0.0, float(np.expm1(pred_log)))
        return round(raw_pred, 2)
