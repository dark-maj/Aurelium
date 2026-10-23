"""
Aurelium - Phase 2: Citation Prediction Models & Evaluation
Trains LightGBM and PyTorch MLP models to forecast 3-year citation impact.
Enforces strict temporal splits (Train: 2015-2020, Val: 2021, Test: 2022-2023)
and absolute leakage barriers (counts_by_year isolated to label construction).
"""

import argparse
import datetime
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Set global reproducibility seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Extensibility hooks for future phases (Phase 5 Novelty, authors, venues)
OPTIONAL_FEATURE_HOOKS = ["author_count", "venue", "novelty_score"]


def extract_metadata_features(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Extract non-leaking metadata features from paper records.
    Strictly excludes counts_by_year and any forward-looking citation metrics.
    """
    rows = []
    for r in records:
        # Referenced works count
        refs = r.get("referenced_works") or []
        n_refs = len(refs)

        # Text lengths
        title = r.get("title") or ""
        abstract = r.get("abstract") or ""
        title_len = len(title)
        abstract_len = len(abstract)
        has_abstract = 1.0 if abstract else 0.0

        # Publication month from ISO date string (YYYY-MM-DD)
        pub_date = r.get("publication_date") or ""
        pub_month = 0
        if len(pub_date) >= 7 and pub_date[4] == "-":
            try:
                pub_month = int(pub_date[5:7])
            except ValueError:
                pub_month = 0

        # Categoricals
        doc_type = str(r.get("type") or "unknown")
        field = str(r.get("field") or "unknown")
        subfield = str(r.get("subfield") or "unknown")
        topic = str(r.get("topic") or "unknown")

        row = {
            "n_referenced_works": n_refs,
            "title_char_len": title_len,
            "abstract_char_len": abstract_len,
            "has_abstract": has_abstract,
            "publication_month": pub_month,
            "type": doc_type,
            "field": field,
            "subfield": subfield,
            "topic": topic,
        }

        # Check optional hooks for future features
        for hook in OPTIONAL_FEATURE_HOOKS:
            if hook in r:
                row[hook] = r[hook]

        rows.append(row)

    return pd.DataFrame(rows)


def build_preprocessor(df_train: pd.DataFrame) -> Tuple[ColumnTransformer, List[str]]:
    """
    Build and fit preprocessor on training data only.
    StandardScales numeric columns and OneHotEncodes categorical columns.
    """
    num_cols = ["n_referenced_works", "title_char_len", "abstract_char_len", "has_abstract", "publication_month"]
    cat_cols = ["type", "field", "subfield", "topic"]

    # Include any optional hook columns present
    for hook in OPTIONAL_FEATURE_HOOKS:
        if hook in df_train.columns:
            if pd.api.types.is_numeric_dtype(df_train[hook]):
                num_cols.append(hook)
            else:
                cat_cols.append(hook)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    preprocessor.fit(df_train)

    # Reconstruct feature names
    feature_names = list(num_cols)
    cat_encoder = preprocessor.named_transformers_["cat"]
    if hasattr(cat_encoder, "get_feature_names_out"):
        feature_names.extend(cat_encoder.get_feature_names_out(cat_cols))

    return preprocessor, feature_names


class PyTorchMLP(nn.Module):
    """Small PyTorch MLP baseline with BatchNorm and Dropout."""

    def __init__(self, in_features: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 8,
) -> PyTorchMLP:
    """Train PyTorch MLP with validation-based early stopping."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PyTorchMLP(X_train.shape[1]).to(device)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_weights = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_weights:
        model.load_state_dict(best_weights)
    model.eval()
    return model


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_k_pct: float = 0.10,
) -> Dict[str, float]:
    """
    Compute evaluation metrics on log-scale targets:
    - Spearman rank correlation (primary)
    - Mean Absolute Error (MAE)
    - Precision@top-10% (fraction of top 10% actual papers captured in top 10% predicted)
    """
    if len(y_true) < 2:
        return {"spearman": 0.0, "mae": 0.0, "precision_top10": 0.0}

    # Spearman rank correlation
    rho, _ = stats.spearmanr(y_true, y_pred)
    if np.isnan(rho):
        rho = 0.0

    # MAE on log1p(citations)
    mae = float(mean_absolute_error(y_true, y_pred))

    # Precision@top-10%
    n_top = max(1, int(len(y_true) * top_k_pct))
    top_actual_idx = set(np.argsort(y_true)[-n_top:])
    top_pred_idx = set(np.argsort(y_pred)[-n_top:])
    overlap = len(top_actual_idx.intersection(top_pred_idx))
    prec_top10 = float(overlap / n_top)

    return {
        "spearman": float(rho),
        "mae": float(mae),
        "precision_top10": prec_top10,
    }


def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, out_path: str):
    """Generate and save predicted vs actual scatter plot with y=x identity line."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.35, edgecolors="none", s=25, color="#1f77b4")

    # Identity reference line
    min_v = min(float(y_true.min()), float(y_pred.min()))
    max_v = max(float(y_true.max()), float(y_pred.max()))
    plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1.5, label="Perfect Alignment (y = x)")

    plt.xlabel("Actual log1p(3-Year Citations)", fontsize=11)
    plt.ylabel("Predicted log1p(3-Year Citations)", fontsize=11)
    plt.title("Predicted vs. Actual Citation Impact (Test Set)", fontsize=12, fontweight="bold")
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_feature_importance(importances: np.ndarray, feature_names: List[str], out_path: str, top_n: int = 20):
    """Plot top N feature importances from LightGBM model."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    indices = np.argsort(importances)[-top_n:]
    top_names = [feature_names[i] if i < len(feature_names) else f"feature_{i}" for i in indices]
    top_vals = importances[indices]

    plt.figure(figsize=(8, 6))
    y_pos = np.arange(len(top_names))
    plt.barh(y_pos, top_vals, align="center", color="#2ca02c", alpha=0.85)
    plt.yticks(y_pos, top_names, fontsize=9)
    plt.xlabel("Split Feature Importance (LightGBM)", fontsize=11)
    plt.title(f"Top {top_n} Most Predictive Citation Features", fontsize=12, fontweight="bold")
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def generate_synthetic_benchmark_records(n_records: int = 500) -> List[Dict[str, Any]]:
    """
    Generate synthetic AI paper records covering 2015-2023 for fast local test mode.
    Guarantees fast end-to-end execution (< 2 minutes) without internet dependencies.
    """
    types = ["article", "conference-paper", "preprint"]
    topics = [
        "Advanced Graph Neural Networks",
        "Transformer Architectures in Computer Vision",
        "Natural Language Processing Techniques",
        "Reinforcement Learning for Robotics",
        "Privacy-Preserving Deep Learning",
    ]

    records = []
    for i in range(n_records):
        year = 2015 + (i % 9)  # 2015..2023 evenly distributed
        month = 1 + (i % 12)
        doc_type = types[i % len(types)]
        topic = topics[i % len(topics)]
        n_refs = random.randint(5, 50)

        # Baseline citations with realistic log-normal distribution
        raw_citations = int(np.random.lognormal(mean=1.5, sigma=1.0))
        # Distribute over 3-year window
        c0 = int(raw_citations * random.uniform(0.1, 0.3))
        c1 = int(raw_citations * random.uniform(0.3, 0.5))
        c2 = raw_citations - c0 - c1

        counts_by_year = [
            {"year": year, "cited_by_count": c0},
            {"year": year + 1, "cited_by_count": c1},
            {"year": year + 2, "cited_by_count": c2},
        ]

        title = f"Empirical Study on {topic} for Scalable Optimization #{i}"
        abstract = f"In this work, we propose a novel approach to {topic.lower()}. We evaluate on benchmarks and show strong performance."

        records.append({
            "id": f"https://openalex.org/W_TEST_{i}",
            "title": title,
            "abstract": abstract,
            "publication_date": f"{year}-{month:02d}-15",
            "publication_year": year,
            "type": doc_type,
            "language": "en",
            "referenced_works": [f"W_REF_{j}" for j in range(n_refs)],
            "counts_by_year": counts_by_year,
            "topic": topic,
            "subfield": "Artificial Intelligence",
            "field": "Computer Science",
        })

    return records


def run_training_pipeline(
    records: List[Dict[str, Any]],
    embeddings: np.ndarray,
    out_models_dir: str = "models/phase2",
    out_reports_dir: str = "reports",
    use_pca_comparison: bool = True,
) -> Dict[str, Any]:
    """
    Execute full Phase 2 training and evaluation pipeline:
    1. Extract labels (y, log1p(y))
    2. Extract metadata features and fit preprocessor on Train only
    3. Train Median/Mean baselines, LightGBM (Full + PCA), and PyTorch MLP
    4. Evaluate on Val and Test splits (overall and per-cohort)
    5. Save models, metadata.json, plots, and markdown report
    """
    from src.citation_labels import build_labels

    os.makedirs(out_models_dir, exist_ok=True)
    figures_dir = os.path.join(out_reports_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print(f"--- Running Citation Prediction Pipeline ({len(records)} records) ---")

    # 1. Labels
    labels_df = build_labels(records)
    y_all = labels_df["log_citations_3yr"].values
    years = labels_df["publication_year"].values

    # 2. Metadata features
    meta_df = extract_metadata_features(records)

    # 3. Temporal splits (Train: 2015-2020, Val: 2021, Test: 2022-2023)
    train_mask = (years >= 2015) & (years <= 2020)
    val_mask = (years == 2021)
    test_mask = (years >= 2022) & (years <= 2023)

    # Fallback for small test datasets where years might not span full range
    if not np.any(train_mask) or not np.any(val_mask) or not np.any(test_mask):
        print("Notice: Dataset does not span standard 2015-2023 years. Using 60/20/20 time split fallback.")
        sorted_years = sorted(np.unique(years))
        n_y = len(sorted_years)
        t_end = sorted_years[max(0, int(n_y * 0.6))]
        v_end = sorted_years[max(1, int(n_y * 0.8))]
        train_mask = years <= t_end
        val_mask = (years > t_end) & (years <= v_end)
        test_mask = years > v_end

    print(f"Split sizes: Train={train_mask.sum()}, Val={val_mask.sum()}, Test={test_mask.sum()}")

    # 4. Preprocess metadata (Strictly fit on Train only)
    preprocessor, meta_feature_names = build_preprocessor(meta_df[train_mask])
    X_meta = preprocessor.transform(meta_df)

    # Combined feature matrix: [SPECTER embeddings (768-d) || Metadata features]
    X_all = np.hstack([embeddings, X_meta])
    feature_names = [f"specter_{i}" for i in range(embeddings.shape[1])] + meta_feature_names

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    # Save preprocessor artifact
    preprocessor_path = os.path.join(out_models_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)

    results: Dict[str, Any] = {}

    # --- MODEL 1: Baselines ---
    median_val = float(np.median(y_train))
    mean_val = float(np.mean(y_train))

    pred_val_median = np.full_like(y_val, fill_value=median_val)
    pred_test_median = np.full_like(y_test, fill_value=median_val)
    results["baseline_median"] = {
        "val": evaluate_predictions(y_val, pred_val_median),
        "test": evaluate_predictions(y_test, pred_test_median),
    }

    # --- MODEL 2: LightGBM (Main Model) ---
    print("Training Main Model: LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=5 if len(X_train) < 500 else 20,
        random_state=SEED,
        verbosity=-1,
    )
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
    )

    pred_val_lgb = lgb_model.predict(X_val)
    pred_test_lgb = lgb_model.predict(X_test)
    results["lightgbm"] = {
        "val": evaluate_predictions(y_val, pred_val_lgb),
        "test": evaluate_predictions(y_test, pred_test_lgb),
    }

    # Save LightGBM booster
    lgb_save_path = os.path.join(out_models_dir, "lightgbm_model.txt")
    lgb_model.booster_.save_model(lgb_save_path)

    # --- MODEL 2b: PCA Comparison (LightGBM on PCA-reduced embeddings + metadata) ---
    if use_pca_comparison and embeddings.shape[1] >= 64:
        print("Training Comparison: LightGBM with PCA on Embeddings...")
        pca_dim = min(64, embeddings.shape[0], embeddings.shape[1])
        pca = PCA(n_components=pca_dim, random_state=SEED)
        pca.fit(embeddings[train_mask])  # Fit on train only

        X_emb_pca = pca.transform(embeddings)
        X_pca_all = np.hstack([X_emb_pca, X_meta])

        lgb_pca = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=5 if len(X_train) < 500 else 20,
            random_state=SEED,
            verbosity=-1,
        )
        lgb_pca.fit(
            X_pca_all[train_mask],
            y_train,
            eval_set=[(X_pca_all[val_mask], y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
        )

        pred_val_pca = lgb_pca.predict(X_pca_all[val_mask])
        pred_test_pca = lgb_pca.predict(X_pca_all[test_mask])
        results["lightgbm_pca"] = {
            "val": evaluate_predictions(y_val, pred_val_pca),
            "test": evaluate_predictions(y_test, pred_test_pca),
        }

    # --- MODEL 3: PyTorch MLP ---
    print("Training Comparison: PyTorch MLP...")
    mlp_model = train_mlp(X_train, y_train, X_val, y_val, epochs=30, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        pred_val_mlp = mlp_model(torch.tensor(X_val, dtype=torch.float32).to(device)).cpu().numpy()
        pred_test_mlp = mlp_model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

    results["pytorch_mlp"] = {
        "val": evaluate_predictions(y_val, pred_val_mlp),
        "test": evaluate_predictions(y_test, pred_test_mlp),
    }

    # Save MLP weights
    torch.save(mlp_model.state_dict(), os.path.join(out_models_dir, "mlp_weights.pt"))

    # --- Cohort Breakdown (Test Set) ---
    test_years = years[test_mask]
    cohort_results: Dict[int, Dict[str, float]] = {}
    for yr in sorted(np.unique(test_years)):
        yr_idx = np.where(test_years == yr)[0]
        if len(yr_idx) >= 2:
            cohort_results[int(yr)] = evaluate_predictions(y_test[yr_idx], pred_test_lgb[yr_idx])

    # --- Figures ---
    scatter_path = os.path.join(figures_dir, "predicted_vs_actual.png")
    plot_predicted_vs_actual(y_test, pred_test_lgb, scatter_path)

    importance_path = os.path.join(figures_dir, "feature_importance.png")
    importances = lgb_model.feature_importances_
    plot_feature_importance(importances, feature_names, importance_path)

    # --- Metadata JSON ---
    metadata = {
        "phase": 2,
        "task": "citation_prediction",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed": SEED,
        "splits": {
            "train_years": [int(y) for y in np.unique(years[train_mask])],
            "val_years": [int(y) for y in np.unique(years[val_mask])],
            "test_years": [int(y) for y in np.unique(years[test_mask])],
            "train_size": int(train_mask.sum()),
            "val_size": int(val_mask.sum()),
            "test_size": int(test_mask.sum()),
        },
        "features": {
            "specter_dim": embeddings.shape[1],
            "metadata_features": meta_feature_names,
            "total_dim": X_all.shape[1],
        },
        "metrics": results,
        "cohort_metrics_test": cohort_results,
        "environment": {
            "python": sys.version,
            "lightgbm": lgb.__version__,
            "torch": torch.__version__,
        },
    }
    with open(os.path.join(out_models_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # --- Markdown Report ---
    report_path = os.path.join(out_reports_dir, "phase2_citation_report.md")
    write_citation_report(report_path, metadata, results, cohort_results)
    print(f"Saved Phase 2 evaluation report to {report_path}")

    return metadata


def write_citation_report(
    out_md: str,
    metadata: Dict[str, Any],
    results: Dict[str, Any],
    cohort_results: Dict[int, Dict[str, float]],
):
    """Write comprehensive markdown evaluation report."""
    os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)

    lines = [
        "# Aurelium - Phase 2: Citation Prediction Evaluation Report",
        "",
        f"**Generated At (UTC)**: `{metadata.get('created_at')}`  ",
        f"**Random Seed**: `{metadata.get('seed')}`  ",
        f"**Splits**: Train={metadata['splits']['train_size']}, Val={metadata['splits']['val_size']}, Test={metadata['splits']['test_size']}  ",
        "",
        "---",
        "",
        "## 1. Overall Model Performance",
        "",
        "Evaluation target is $\\log(1 + y)$, where $y$ is the 3-year citation count from publication year.",
        "",
        "| Model | Val Spearman ($\\rho$) | Val MAE | Val Prec@Top-10% | Test Spearman ($\\rho$) | Test MAE | Test Prec@Top-10% |",
        "|---|---|---|---|---|---|---|",
    ]

    for model_name, metrics in results.items():
        v = metrics.get("val", {})
        t = metrics.get("test", {})
        display_name = {
            "baseline_median": "Baseline (Median)",
            "lightgbm": "**LightGBM (Main)**",
            "lightgbm_pca": "LightGBM + PCA",
            "pytorch_mlp": "PyTorch MLP",
        }.get(model_name, model_name)

        lines.append(
            f"| {display_name} | {v.get('spearman', 0.0):.4f} | {v.get('mae', 0.0):.4f} | {v.get('precision_top10', 0.0)*100:.1f}% | "
            f"{t.get('spearman', 0.0):.4f} | {t.get('mae', 0.0):.4f} | {t.get('precision_top10', 0.0)*100:.1f}% |"
        )

    lines.extend([
        "",
        "## 2. Test Performance by Publication Year Cohort",
        "",
        "| Cohort Year | Spearman Correlation ($\\rho$) | MAE (log scale) | Precision@Top-10% |",
        "|---|---|---|---|",
    ])
    for yr, m in cohort_results.items():
        lines.append(f"| {yr} | {m.get('spearman', 0.0):.4f} | {m.get('mae', 0.0):.4f} | {m.get('precision_top10', 0.0)*100:.1f}% |")

    lines.extend([
        "",
        "## 3. Visualizations",
        "",
        "- **Predicted vs. Actual Alignment**: `reports/figures/predicted_vs_actual.png`",
        "- **Top Predictive Feature Importance**: `reports/figures/feature_importance.png`",
        "",
        "## 4. Key Findings & Limitations",
        "",
        "1. **Log Transformation Efficacy**: Compressed power-law citation skew, ensuring gradient descent stability.",
        "2. **Strict Temporal Isolation**: Scalers and encoders fit solely on historical training cohorts.",
        "3. **Intrinsic Noise**: Early citations are partially stochastic (author prestige and networking are unobserved in title/abstract).",
    ])

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    """CLI entry point for Phase 2 citation modeling."""
    parser = argparse.ArgumentParser(description="Aurelium Citation Model Training")
    parser.add_argument("--jsonl", default="data/raw/papers.jsonl", help="Path to papers.jsonl")
    parser.add_argument("--embeddings", default="data/clean/embeddings.npy", help="Path to embeddings.npy")
    parser.add_argument("--models-dir", default="models/phase2", help="Directory to save model artifacts")
    parser.add_argument("--reports-dir", default="reports", help="Directory to save report and plots")
    parser.add_argument("--test", action="store_true", help="Run fast end-to-end benchmark on 500 synthetic records (< 2 min)")
    args = parser.parse_args()

    if args.test:
        print("=== Running Phase 2 Fast Benchmark (--test mode) ===")
        records = generate_synthetic_benchmark_records(n_records=500)
        # Generate temporary embeddings
        from src.embed import embed_corpus
        test_npy = os.path.join(args.models_dir, "test_embeddings.npy")
        test_ids = os.path.join(args.models_dir, "test_ids.json")
        embeddings, ids, _ = embed_corpus(records, cache_path=test_npy, ids_path=test_ids, batch_size=32)

        run_training_pipeline(
            records=records,
            embeddings=embeddings,
            out_models_dir=args.models_dir,
            out_reports_dir=args.reports_dir,
            use_pca_comparison=True,
        )
        print("\nPhase 2 test run completed successfully!")
    else:
        if not os.path.exists(args.jsonl) or not os.path.exists(args.embeddings):
            print(f"Error: Missing inputs ({args.jsonl} or {args.embeddings}).")
            print("Use --test to run the fast verification benchmark.")
            return

        from src.citation_labels import stream_jsonl
        records = list(stream_jsonl(args.jsonl))
        embeddings = np.load(args.embeddings)

        run_training_pipeline(
            records=records,
            embeddings=embeddings,
            out_models_dir=args.models_dir,
            out_reports_dir=args.reports_dir,
            use_pca_comparison=True,
        )


if __name__ == "__main__":
    main()
