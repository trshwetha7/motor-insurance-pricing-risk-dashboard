from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.train_model import MODEL_FEATURES
from src.utils import (
    BEST_MODEL_PATH,
    DASHBOARD_KPI_PATH,
    FEATURE_IMPORTANCE_PATH,
    FIGURES_DIR,
    HOLDOUT_PREDICTIONS_PATH,
    MODEL_METRICS_PATH,
    PRICING_ADEQUACY_PATH,
    RANDOM_STATE,
    RISK_DECILE_SUMMARY_PATH,
    TRAINING_METADATA_PATH,
    ensure_project_dirs,
    load_json,
    load_processed_data,
    save_json,
)

os.environ.setdefault("MPLCONFIGDIR", str(FIGURES_DIR.parent / ".mplconfig"))
import matplotlib.pyplot as plt


def _save_calibration_chart(decile_summary: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        decile_summary["risk_decile"],
        decile_summary["actual_pure_premium"],
        marker="o",
        linewidth=2.5,
        label="Actual pure premium",
    )
    ax.plot(
        decile_summary["risk_decile"],
        decile_summary["predicted_pure_premium"],
        marker="o",
        linewidth=2.5,
        label="Predicted pure premium",
    )
    ax.set_title("Calibration by Risk Decile")
    ax.set_xlabel("Predicted pure premium decile")
    ax.set_ylabel("Pure premium")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_by_decile.png", dpi=200)
    plt.close(fig)


def _save_lift_chart(decile_summary: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(decile_summary))
    width = 0.38
    ax.bar(
        x - width / 2,
        decile_summary["actual_lift_vs_portfolio"],
        width=width,
        label="Actual lift",
    )
    ax.bar(
        x + width / 2,
        decile_summary["predicted_lift_vs_portfolio"],
        width=width,
        label="Predicted lift",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(decile_summary["risk_decile"], rotation=25)
    ax.set_title("Lift Chart by Risk Decile")
    ax.set_xlabel("Predicted pure premium decile")
    ax.set_ylabel("Lift vs portfolio average")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "lift_chart.png", dpi=200)
    plt.close(fig)


def _save_feature_importance_chart(feature_importance_df: pd.DataFrame) -> None:
    display_df = feature_importance_df.head(12).sort_values("importance", ascending=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(display_df["feature"], display_df["importance"], color="#0F766E")
    ax.set_title("Permutation Importance")
    ax.set_xlabel("Mean importance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=200)
    plt.close(fig)


def run_evaluation() -> dict[str, pd.DataFrame | dict[str, float]]:
    ensure_project_dirs()
    if not HOLDOUT_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "Holdout predictions not found. Run `python src/train_model.py` first."
        )

    holdout_predictions = pd.read_csv(HOLDOUT_PREDICTIONS_PATH)
    decile_summary = pd.read_csv(RISK_DECILE_SUMMARY_PATH)
    metrics_df = pd.read_csv(MODEL_METRICS_PATH)
    pricing_summary = pd.read_csv(PRICING_ADEQUACY_PATH)
    training_metadata = joblib.load(TRAINING_METADATA_PATH)
    best_model = joblib.load(BEST_MODEL_PATH)

    df = load_processed_data()
    X = df[MODEL_FEATURES]
    y = df["pure_premium"]
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    sample_size = min(8_000, len(X_test))
    sample_index = X_test.sample(sample_size, random_state=RANDOM_STATE).index
    permutation = permutation_importance(
        best_model,
        X_test.loc[sample_index],
        y_test.loc[sample_index],
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="neg_root_mean_squared_error",
    )
    feature_importance_df = (
        pd.DataFrame(
            {
                "feature": MODEL_FEATURES,
                "importance": permutation.importances_mean,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    feature_importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    _save_calibration_chart(decile_summary)
    _save_lift_chart(decile_summary)
    _save_feature_importance_chart(feature_importance_df)

    existing_kpis = load_json(DASHBOARD_KPI_PATH)
    underpriced_share = float(existing_kpis["underpriced_share"])
    high_risk_decile = (
        decile_summary.sort_values("predicted_pure_premium", ascending=False)
        .iloc[0]["risk_decile"]
    )

    dashboard_kpis = existing_kpis
    dashboard_kpis.update(
        {
            "underpriced_share": underpriced_share,
            "highest_risk_decile": high_risk_decile,
            "top_feature": feature_importance_df.iloc[0]["feature"],
            "top_feature_importance": float(feature_importance_df.iloc[0]["importance"]),
            "best_model_name": training_metadata["best_model_name"],
        }
    )
    save_json(dashboard_kpis, DASHBOARD_KPI_PATH)

    return {
        "model_metrics": metrics_df,
        "decile_summary": decile_summary,
        "pricing_summary": pricing_summary,
        "feature_importance": feature_importance_df,
        "dashboard_kpis": dashboard_kpis,
    }


def main() -> None:
    results = run_evaluation()
    print(
        "Saved evaluation assets, including charts, using "
        f"{results['dashboard_kpis']['best_model_name']}."
    )


if __name__ == "__main__":
    main()
