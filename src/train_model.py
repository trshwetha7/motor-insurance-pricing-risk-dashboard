from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor, TweedieRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_tweedie_deviance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_prep import build_processed_dataset
from src.utils import (
    BENCHMARK_MODEL_PATH,
    BEST_MODEL_PATH,
    DASHBOARD_KPI_PATH,
    FEATURE_IMPORTANCE_PATH,
    FREQUENCY_MODEL_PATH,
    HOLDOUT_PREDICTIONS_PATH,
    MODEL_METRICS_PATH,
    PRICING_ADEQUACY_PATH,
    PROCESSED_PARQUET_PATH,
    PROCESSED_CSV_PATH,
    RANDOM_STATE,
    RISK_DECILE_SUMMARY_PATH,
    RISK_LOADING,
    TRAINING_METADATA_PATH,
    assign_prediction_deciles,
    ensure_project_dirs,
    load_processed_data,
    pricing_adequacy_label,
    save_json,
)


MODEL_FEATURES = [
    "Area",
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "VehBrand",
    "VehGas",
    "Density",
    "Region",
    "Exposure",
    "driver_age_band",
    "vehicle_age_band",
    "bonus_malus_band",
    "density_band",
]

CATEGORICAL_FEATURES = [
    "Area",
    "VehBrand",
    "VehGas",
    "Region",
    "driver_age_band",
    "vehicle_age_band",
    "bonus_malus_band",
    "density_band",
]

NUMERIC_FEATURES = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density", "Exposure"]


@dataclass
class ModelArtifact:
    name: str
    pipeline: Pipeline
    predictions: np.ndarray


def build_glm_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def build_tree_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )


def evaluate_predictions(
    y_true: pd.Series, y_pred: np.ndarray, sample_weight: pd.Series | None = None
) -> dict[str, float]:
    clipped_predictions = np.clip(y_pred, 1e-6, None)
    return {
        "MAE": mean_absolute_error(y_true, clipped_predictions, sample_weight=sample_weight),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    y_true, clipped_predictions, sample_weight=sample_weight
                )
            )
        ),
        "Mean Tweedie Deviance": mean_tweedie_deviance(
            y_true, clipped_predictions, power=1.5, sample_weight=sample_weight
        ),
    }


def run_training() -> dict[str, pd.DataFrame | dict[str, float] | str]:
    ensure_project_dirs()
    try:
        df = load_processed_data()
    except FileNotFoundError:
        df = build_processed_dataset()

    X = df[MODEL_FEATURES].copy()
    y_pure = df["pure_premium"].copy()
    y_frequency = df["claim_frequency"].copy()
    exposure = df["Exposure"].copy()

    (
        X_train,
        X_test,
        y_pure_train,
        y_pure_test,
        y_freq_train,
        y_freq_test,
        exposure_train,
        exposure_test,
        df_train,
        df_test,
    ) = train_test_split(
        X,
        y_pure,
        y_frequency,
        exposure,
        df,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    tweedie_model = Pipeline(
        steps=[
            ("preprocessor", build_glm_preprocessor()),
            (
                "model",
                TweedieRegressor(power=1.5, alpha=1e-4, link="log", max_iter=1_000),
            ),
        ]
    )
    tweedie_model.fit(X_train, y_pure_train, model__sample_weight=exposure_train)

    frequency_model = Pipeline(
        steps=[
            ("preprocessor", build_glm_preprocessor()),
            (
                "model",
                PoissonRegressor(alpha=1e-4, max_iter=1_000),
            ),
        ]
    )
    frequency_model.fit(X_train, y_freq_train, model__sample_weight=exposure_train)

    benchmark_model = Pipeline(
        steps=[
            ("preprocessor", build_tree_preprocessor()),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="poisson",
                    learning_rate=0.05,
                    max_depth=6,
                    min_samples_leaf=200,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    benchmark_model.fit(X_train, y_pure_train, model__sample_weight=exposure_train)

    pure_model_artifacts = [
        ModelArtifact("TweedieRegressor", tweedie_model, tweedie_model.predict(X_test)),
        ModelArtifact(
            "HistGradientBoostingRegressor",
            benchmark_model,
            benchmark_model.predict(X_test),
        ),
    ]

    pure_metrics_records: list[dict[str, float | str]] = []
    best_name = ""
    best_model: Pipeline | None = None
    best_rmse = float("inf")

    for artifact in pure_model_artifacts:
        metrics = evaluate_predictions(y_pure_test, artifact.predictions, exposure_test)
        pure_metrics_records.append({"model": artifact.name, **metrics})
        if metrics["RMSE"] < best_rmse:
            best_rmse = metrics["RMSE"]
            best_name = artifact.name
            best_model = artifact.pipeline

    frequency_predictions = np.clip(frequency_model.predict(X_test), 0, None)
    frequency_metrics = evaluate_predictions(
        y_freq_test,
        frequency_predictions,
        exposure_test,
    )
    pure_metrics_records.append({"model": "PoissonRegressor (frequency)", **frequency_metrics})

    assert best_model is not None

    best_predictions = np.clip(best_model.predict(X_test), 0, None)
    train_predictions_for_edges = np.clip(best_model.predict(X_train), 0, None)
    deciles, edges = assign_prediction_deciles(best_predictions, edges=None, n_bins=10)

    portfolio_avg_pure_premium = float(np.average(y_pure_train, weights=exposure_train))
    baseline_premium = portfolio_avg_pure_premium * (1 + RISK_LOADING)
    baseline_premiums = np.repeat(baseline_premium, len(X_test))

    adequacy_labels: list[str] = []
    pricing_recommendations: list[str] = []
    for prediction in best_predictions:
        label, recommendation = pricing_adequacy_label(prediction, baseline_premium)
        adequacy_labels.append(label)
        pricing_recommendations.append(recommendation)

    holdout_predictions = df_test.reset_index(drop=True).copy()
    holdout_predictions["predicted_claim_frequency"] = frequency_predictions
    holdout_predictions["predicted_pure_premium"] = best_predictions
    holdout_predictions["baseline_premium"] = baseline_premiums
    holdout_predictions["risk_decile"] = deciles.values
    holdout_predictions["pricing_adequacy"] = adequacy_labels
    holdout_predictions["pricing_recommendation"] = pricing_recommendations
    holdout_predictions["rate_gap"] = (
        holdout_predictions["predicted_pure_premium"] - holdout_predictions["baseline_premium"]
    )
    holdout_predictions["rate_gap_pct"] = (
        holdout_predictions["rate_gap"] / holdout_predictions["baseline_premium"]
    )
    risk_summary = (
        holdout_predictions.groupby("risk_decile", observed=False)
        .agg(
            policies=("IDpol", "count"),
            exposure=("Exposure", "sum"),
            actual_pure_premium=("pure_premium", "mean"),
            predicted_pure_premium=("predicted_pure_premium", "mean"),
            baseline_premium=("baseline_premium", "mean"),
            avg_frequency=("claim_frequency", "mean"),
        )
        .reset_index()
    )
    risk_summary["actual_lift_vs_portfolio"] = (
        risk_summary["actual_pure_premium"] / portfolio_avg_pure_premium
    )
    risk_summary["predicted_lift_vs_portfolio"] = (
        risk_summary["predicted_pure_premium"] / portfolio_avg_pure_premium
    )
    risk_summary.to_csv(RISK_DECILE_SUMMARY_PATH, index=False)

    pricing_summary = (
        holdout_predictions.groupby("pricing_adequacy", observed=False)
        .agg(
            policies=("IDpol", "count"),
            avg_predicted_pure_premium=("predicted_pure_premium", "mean"),
            avg_baseline_premium=("baseline_premium", "mean"),
            avg_rate_gap_pct=("rate_gap_pct", "mean"),
            avg_claim_frequency=("claim_frequency", "mean"),
        )
        .reset_index()
        .sort_values("avg_predicted_pure_premium", ascending=False)
    )
    pricing_summary.to_csv(PRICING_ADEQUACY_PATH, index=False)

    dashboard_policy_sample = holdout_predictions.sample(
        n=min(25_000, len(holdout_predictions)),
        random_state=RANDOM_STATE,
    ).sort_values("predicted_pure_premium")
    dashboard_policy_sample.to_csv(HOLDOUT_PREDICTIONS_PATH, index=False)

    metrics_df = pd.DataFrame(pure_metrics_records)
    metrics_df.to_csv(MODEL_METRICS_PATH, index=False)

    training_metadata = {
        "best_model_name": best_name,
        "model_features": MODEL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "portfolio_avg_pure_premium": portfolio_avg_pure_premium,
        "risk_loading": RISK_LOADING,
        "baseline_premium": baseline_premium,
        "risk_decile_edges": edges,
        "default_vehicle_brand": str(df["VehBrand"].mode().iloc[0]),
        "training_rows": int(df_train.shape[0]),
        "holdout_rows": int(df_test.shape[0]),
        "dashboard_sample_rows": int(dashboard_policy_sample.shape[0]),
        "processed_data_path": str(
            PROCESSED_PARQUET_PATH
            if PROCESSED_PARQUET_PATH.exists()
            else PROCESSED_CSV_PATH
        ),
        "train_prediction_quantiles": np.quantile(
            train_predictions_for_edges, q=np.linspace(0, 1, 11)
        ).tolist(),
    }

    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(frequency_model, FREQUENCY_MODEL_PATH)
    joblib.dump(benchmark_model, BENCHMARK_MODEL_PATH)
    joblib.dump(training_metadata, TRAINING_METADATA_PATH)

    save_json(
        {
            "best_model_name": best_name,
            "portfolio_avg_pure_premium": portfolio_avg_pure_premium,
            "baseline_premium": baseline_premium,
            "best_model_rmse": best_rmse,
            "holdout_policy_count": int(holdout_predictions.shape[0]),
            "underpriced_share": float(
                (holdout_predictions["pricing_adequacy"] == "Underpriced / high expected loss")
                .mean()
            ),
        },
        DASHBOARD_KPI_PATH,
    )

    feature_stub = pd.DataFrame({"feature": MODEL_FEATURES, "importance": np.nan})
    feature_stub.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    return {
        "model_metrics": metrics_df,
        "risk_summary": risk_summary,
        "pricing_summary": pricing_summary,
        "best_model_name": best_name,
    }


def main() -> None:
    results = run_training()
    print(f"Saved model artifacts. Best pure premium model: {results['best_model_name']}")


if __name__ == "__main__":
    main()
