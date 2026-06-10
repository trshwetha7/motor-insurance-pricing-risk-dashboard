from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

PROCESSED_PARQUET_PATH = PROCESSED_DIR / "freMTPL2_pricing.parquet"
PROCESSED_CSV_PATH = PROCESSED_DIR / "freMTPL2_pricing.csv"
METADATA_JSON_PATH = PROCESSED_DIR / "dataset_metadata.json"

MODEL_METRICS_PATH = REPORTS_DIR / "model_metrics.csv"
RISK_DECILE_SUMMARY_PATH = REPORTS_DIR / "risk_decile_summary.csv"
PRICING_ADEQUACY_PATH = REPORTS_DIR / "pricing_adequacy_summary.csv"
HOLDOUT_PREDICTIONS_PATH = REPORTS_DIR / "holdout_predictions.csv"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
DASHBOARD_KPI_PATH = REPORTS_DIR / "dashboard_kpis.json"

BEST_MODEL_PATH = MODELS_DIR / "best_pure_premium_model.joblib"
FREQUENCY_MODEL_PATH = MODELS_DIR / "frequency_model.joblib"
BENCHMARK_MODEL_PATH = MODELS_DIR / "tree_benchmark_model.joblib"
TRAINING_METADATA_PATH = MODELS_DIR / "training_metadata.joblib"

RANDOM_STATE = 42
RISK_LOADING = 0.15


def ensure_project_dirs() -> None:
    """Create project folders if they do not exist."""
    for directory in (APP_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_processed_data(df: pd.DataFrame) -> Path:
    """Persist the processed dataset with a CSV fallback."""
    ensure_project_dirs()
    try:
        df.to_parquet(PROCESSED_PARQUET_PATH, index=False)
        return PROCESSED_PARQUET_PATH
    except Exception:
        df.to_csv(PROCESSED_CSV_PATH, index=False)
        return PROCESSED_CSV_PATH


def load_processed_data() -> pd.DataFrame:
    if PROCESSED_PARQUET_PATH.exists():
        return pd.read_parquet(PROCESSED_PARQUET_PATH)
    if PROCESSED_CSV_PATH.exists():
        return pd.read_csv(PROCESSED_CSV_PATH)
    raise FileNotFoundError(
        "Processed dataset not found. Run `python src/data_prep.py` first."
    )


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.clip(lower=1e-6)


def build_density_bands(series: pd.Series) -> pd.Series:
    labels = ["Rural", "Semi-Rural", "Town", "City", "Metro"]
    banded = pd.qcut(series.rank(method="first"), q=5, labels=labels)
    return banded.astype(str)


def assign_prediction_deciles(
    values: pd.Series | np.ndarray,
    edges: list[float] | np.ndarray | None = None,
    n_bins: int = 10,
) -> tuple[pd.Series, list[float]]:
    """Assign deciles using either supplied edges or empirical quantiles."""
    predictions = np.asarray(values, dtype=float)
    if edges is None:
        quantiles = np.quantile(predictions, q=np.linspace(0, 1, n_bins + 1))
        quantiles[0] = -np.inf
        quantiles[-1] = np.inf
        deduped = np.unique(quantiles)
        if deduped.size < 3:
            deduped = np.array([-np.inf, np.inf])
        edges = deduped.tolist()
    else:
        edges = [float(edge) for edge in edges]

    labels = [f"Decile {index}" for index in range(1, len(edges))]
    bins = pd.cut(predictions, bins=edges, labels=labels, include_lowest=True)
    if bins.isna().any():
        fallback = pd.Series(np.repeat(labels[-1], len(predictions)), index=range(len(predictions)))
        bins = bins.astype(object).where(~bins.isna(), fallback)
    return pd.Series(bins, index=range(len(predictions))), edges


def pricing_adequacy_label(
    predicted_pure_premium: float, baseline_premium: float
) -> tuple[str, str]:
    ratio = predicted_pure_premium / max(baseline_premium, 1e-6)
    if ratio >= 1.1:
        return (
            "Underpriced / high expected loss",
            "Increase premium or tighten underwriting appetite for similar risks.",
        )
    if ratio <= 0.9:
        return (
            "Competitive opportunity / lower risk",
            "Maintain competitiveness and consider targeted growth in this segment.",
        )
    return (
        "Adequately priced",
        "Current premium level is broadly aligned with expected loss cost.",
    )


def currency_formatter(value: float) -> str:
    return f"${value:,.0f}"
