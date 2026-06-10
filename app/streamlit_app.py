from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.evaluate import run_evaluation
from src.train_model import MODEL_FEATURES, run_training
from src.utils import (
    BEST_MODEL_PATH,
    DASHBOARD_KPI_PATH,
    FEATURE_IMPORTANCE_PATH,
    FREQUENCY_MODEL_PATH,
    HOLDOUT_PREDICTIONS_PATH,
    MODEL_METRICS_PATH,
    PRICING_ADEQUACY_PATH,
    RISK_DECILE_SUMMARY_PATH,
    TRAINING_METADATA_PATH,
    assign_prediction_deciles,
    currency_formatter,
    load_json,
    pricing_adequacy_label,
)


st.set_page_config(
    page_title="Motor Insurance Pricing Risk Dashboard",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_report_data() -> dict[str, pd.DataFrame | dict]:
    return {
        "kpis": load_json(DASHBOARD_KPI_PATH),
        "metrics": pd.read_csv(MODEL_METRICS_PATH),
        "risk_summary": pd.read_csv(RISK_DECILE_SUMMARY_PATH),
        "pricing_summary": pd.read_csv(PRICING_ADEQUACY_PATH),
        "feature_importance": pd.read_csv(FEATURE_IMPORTANCE_PATH),
        "holdout_predictions": pd.read_csv(HOLDOUT_PREDICTIONS_PATH),
    }


@st.cache_resource(show_spinner=False)
def load_models() -> tuple[object, object, dict]:
    try:
        pure_model = joblib.load(BEST_MODEL_PATH)
        frequency_model = joblib.load(FREQUENCY_MODEL_PATH)
        metadata = joblib.load(TRAINING_METADATA_PATH)
        return pure_model, frequency_model, metadata
    except Exception:
        # Rebuild artifacts if local model files were created with an incompatible sklearn/numpy version.
        run_training()
        run_evaluation()
        pure_model = joblib.load(BEST_MODEL_PATH)
        frequency_model = joblib.load(FREQUENCY_MODEL_PATH)
        metadata = joblib.load(TRAINING_METADATA_PATH)
        return pure_model, frequency_model, metadata


def render_header() -> None:
    st.title("Motor Insurance Pricing & Underwriting Risk Dashboard")
    st.caption(
        "A portfolio view of expected claim cost, pricing adequacy, and policy-level risk scoring."
    )


def executive_overview(report_data: dict[str, pd.DataFrame | dict]) -> None:
    kpis = report_data["kpis"]
    risk_summary = report_data["risk_summary"]
    pricing_summary = report_data["pricing_summary"]

    metric_columns = st.columns(4)
    metric_columns[0].metric("Best Pure Premium Model", str(kpis["best_model_name"]))
    metric_columns[1].metric(
        "Portfolio Avg Pure Premium",
        currency_formatter(float(kpis["portfolio_avg_pure_premium"])),
    )
    metric_columns[2].metric(
        "Baseline Premium",
        currency_formatter(float(kpis["baseline_premium"])),
    )
    metric_columns[3].metric(
        "Underpriced Share",
        f"{float(kpis['underpriced_share']) * 100:.1f}%",
    )

    chart_col, summary_col = st.columns((1.5, 1))
    with chart_col:
        fig = px.bar(
            risk_summary,
            x="risk_decile",
            y="predicted_pure_premium",
            color="actual_lift_vs_portfolio",
            color_continuous_scale="Tealgrn",
            title="Expected Claim Cost by Risk Decile",
            labels={
                "risk_decile": "Risk decile",
                "predicted_pure_premium": "Predicted pure premium",
                "actual_lift_vs_portfolio": "Actual lift",
            },
        )
        fig.update_layout(height=420, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Predicted pure premium rises sharply in the highest deciles, highlighting where underwriting discipline matters most."
        )

    with summary_col:
        st.subheader("Pricing Signals")
        st.dataframe(
            pricing_summary.assign(
                avg_predicted_pure_premium=lambda frame: frame[
                    "avg_predicted_pure_premium"
                ].round(0),
                avg_baseline_premium=lambda frame: frame["avg_baseline_premium"].round(0),
                avg_rate_gap_pct=lambda frame: (frame["avg_rate_gap_pct"] * 100).round(1),
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "The adequacy summary translates model output into actionable premium and appetite decisions."
        )


def risk_segmentation(report_data: dict[str, pd.DataFrame | dict]) -> None:
    risk_summary = report_data["risk_summary"]
    holdout_predictions = report_data["holdout_predictions"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=risk_summary["risk_decile"],
            y=risk_summary["actual_pure_premium"],
            mode="lines+markers",
            name="Actual pure premium",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=risk_summary["risk_decile"],
            y=risk_summary["predicted_pure_premium"],
            mode="lines+markers",
            name="Predicted pure premium",
        )
    )
    fig.update_layout(
        title="Actual vs Predicted Pure Premium by Risk Decile",
        xaxis_title="Risk decile",
        yaxis_title="Pure premium",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Calibration is strongest when actual pure premium tracks closely to the predicted pure premium across deciles."
    )

    mix_fig = px.histogram(
        holdout_predictions,
        x="risk_decile",
        color="pricing_adequacy",
        barmode="stack",
        title="Adequacy Mix Across Risk Deciles",
        labels={"risk_decile": "Risk decile", "count": "Policies"},
        color_discrete_sequence=["#B91C1C", "#0F766E", "#2563EB"],
    )
    mix_fig.update_layout(height=420)
    st.plotly_chart(mix_fig, use_container_width=True)
    st.caption(
        "This view shows whether the riskier parts of the portfolio are more likely to be underpriced."
    )

    st.subheader("Risk Segment Summary")
    st.dataframe(
        risk_summary.assign(
            actual_pure_premium=lambda frame: frame["actual_pure_premium"].round(0),
            predicted_pure_premium=lambda frame: frame["predicted_pure_premium"].round(0),
            baseline_premium=lambda frame: frame["baseline_premium"].round(0),
            actual_lift_vs_portfolio=lambda frame: frame["actual_lift_vs_portfolio"].round(2),
            predicted_lift_vs_portfolio=lambda frame: frame[
                "predicted_lift_vs_portfolio"
            ].round(2),
        ),
        use_container_width=True,
        hide_index=True,
    )


def model_performance(report_data: dict[str, pd.DataFrame | dict]) -> None:
    metrics = report_data["metrics"]
    feature_importance = report_data["feature_importance"].head(10)

    left_col, right_col = st.columns((1, 1))
    with left_col:
        st.subheader("Holdout Metrics")
        st.dataframe(
            metrics.assign(
                MAE=lambda frame: frame["MAE"].round(4),
                RMSE=lambda frame: frame["RMSE"].round(4),
                **{"Mean Tweedie Deviance": lambda frame: frame["Mean Tweedie Deviance"].round(4)},
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "The GLM and tree benchmark are compared on the holdout set using error and Tweedie deviance metrics."
        )

    with right_col:
        fig = px.bar(
            feature_importance.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            title="Top Feature Drivers",
            labels={"importance": "Permutation importance", "feature": "Feature"},
            color="importance",
            color_continuous_scale="Teal",
        )
        fig.update_layout(height=420, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Driver age, vehicle age, prior claims experience, and territory typically dominate expected loss differentiation."
        )


def pricing_adequacy(report_data: dict[str, pd.DataFrame | dict]) -> None:
    pricing_summary = report_data["pricing_summary"]
    holdout_predictions = report_data["holdout_predictions"]

    fig = px.box(
        holdout_predictions,
        x="pricing_adequacy",
        y="predicted_pure_premium",
        color="pricing_adequacy",
        title="Predicted Pure Premium by Pricing Adequacy Outcome",
        color_discrete_sequence=["#B91C1C", "#0F766E", "#2563EB"],
    )
    fig.update_layout(height=420, xaxis_title="Pricing adequacy", yaxis_title="Predicted pure premium")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Underpriced policies concentrate at materially higher expected loss costs than the competitive opportunity pool."
    )

    pricing_display = pricing_summary.copy()
    pricing_display["avg_rate_gap_pct"] = (pricing_display["avg_rate_gap_pct"] * 100).round(1)
    st.subheader("Business Recommendation Summary")
    st.dataframe(pricing_display.round(2), use_container_width=True, hide_index=True)
    st.caption(
        "These labels can support pricing actions, underwriting appetite review, and targeted commercial growth."
    )


def policy_scoring_demo() -> None:
    with st.spinner("Loading scoring models..."):
        pure_model, frequency_model, metadata = load_models()

    st.subheader("Policy Scoring Demo")
    st.caption(
        f"Vehicle brand is held constant at the modal portfolio brand ({metadata['default_vehicle_brand']}) to keep the demo focused on common underwriting inputs."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        driver_age = st.slider("Driver age", min_value=18, max_value=90, value=40)
        vehicle_age = st.slider("Vehicle age", min_value=0, max_value=25, value=6)
        bonus_malus = st.slider("BonusMalus", min_value=50, max_value=200, value=90)
    with col2:
        vehicle_power = st.slider("Vehicle power", min_value=4, max_value=15, value=7)
        vehicle_gas = st.selectbox("Vehicle gas type", ["Regular", "Diesel"])
        area = st.selectbox("Area", ["A", "B", "C", "D", "E", "F"])
    with col3:
        region = st.selectbox(
            "Region",
            ["R11", "R21", "R22", "R23", "R24", "R25", "R26", "R31", "R41", "R42", "R43", "R52", "R53", "R54", "R72", "R73", "R74", "R82", "R83", "R91", "R93", "R94"],
        )
        density = st.number_input("Density", min_value=1, max_value=30_000, value=3_000)
        exposure = st.slider("Exposure", min_value=0.05, max_value=1.0, value=1.0, step=0.05)

    if st.button("Score Policy", type="primary"):
        driver_band = pd.cut(
            pd.Series([driver_age]),
            bins=[17, 25, 35, 50, 65, 100],
            labels=["18-25", "26-35", "36-50", "51-65", "66+"],
            include_lowest=True,
        ).astype(str)[0]
        vehicle_band = pd.cut(
            pd.Series([vehicle_age]),
            bins=[-1, 2, 5, 10, 20, 200],
            labels=["0-2", "3-5", "6-10", "11-20", "21+"],
            include_lowest=True,
        ).astype(str)[0]
        bonus_band = pd.cut(
            pd.Series([bonus_malus]),
            bins=[-1, 75, 100, 125, 150, 1000],
            labels=["<=75", "76-100", "101-125", "126-150", "150+"],
            include_lowest=True,
        ).astype(str)[0]

        if density < 100:
            density_band = "Rural"
        elif density < 500:
            density_band = "Semi-Rural"
        elif density < 2_000:
            density_band = "Town"
        elif density < 8_000:
            density_band = "City"
        else:
            density_band = "Metro"

        scoring_row = pd.DataFrame(
            [
                {
                    "Area": area,
                    "VehPower": vehicle_power,
                    "VehAge": vehicle_age,
                    "DrivAge": driver_age,
                    "BonusMalus": bonus_malus,
                    "VehBrand": metadata["default_vehicle_brand"],
                    "VehGas": vehicle_gas,
                    "Density": density,
                    "Region": region,
                    "Exposure": exposure,
                    "driver_age_band": driver_band,
                    "vehicle_age_band": vehicle_band,
                    "bonus_malus_band": bonus_band,
                    "density_band": density_band,
                }
            ]
        )[MODEL_FEATURES]

        predicted_frequency = float(np.clip(frequency_model.predict(scoring_row)[0], 0, None))
        predicted_pure_premium = float(np.clip(pure_model.predict(scoring_row)[0], 0, None))
        risk_decile = assign_prediction_deciles(
            np.array([predicted_pure_premium]),
            edges=metadata["risk_decile_edges"],
        )[0].iloc[0]
        adequacy_label, recommendation = pricing_adequacy_label(
            predicted_pure_premium,
            float(metadata["baseline_premium"]),
        )

        metric_columns = st.columns(4)
        metric_columns[0].metric("Predicted claim frequency", f"{predicted_frequency:.3f}")
        metric_columns[1].metric(
            "Expected claim cost",
            currency_formatter(predicted_pure_premium),
        )
        metric_columns[2].metric("Risk decile", str(risk_decile))
        metric_columns[3].metric("Adequacy label", adequacy_label)

        st.info(recommendation)


def main() -> None:
    render_header()
    try:
        report_data = load_report_data()
    except FileNotFoundError:
        st.error(
            "Project artifacts are not available yet. Run `python src/data_prep.py`, "
            "`python src/train_model.py`, and `python src/evaluate.py` first."
        )
        return

    page = st.sidebar.radio(
        "Navigate",
        [
            "Executive Overview",
            "Risk Segmentation",
            "Model Performance",
            "Pricing Adequacy",
            "Policy Scoring Demo",
        ],
    )

    if page == "Executive Overview":
        executive_overview(report_data)
    elif page == "Risk Segmentation":
        risk_segmentation(report_data)
    elif page == "Model Performance":
        model_performance(report_data)
    elif page == "Pricing Adequacy":
        pricing_adequacy(report_data)
    else:
        policy_scoring_demo()


if __name__ == "__main__":
    main()
