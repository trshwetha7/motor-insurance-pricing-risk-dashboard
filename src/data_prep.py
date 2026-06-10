from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from sklearn.datasets import fetch_openml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils import (
    DATA_DIR,
    METADATA_JSON_PATH,
    PROCESSED_CSV_PATH,
    ensure_project_dirs,
    save_json,
    save_processed_data,
    safe_divide,
    build_density_bands,
)


FREQ_DATA_ID = 41214
SEV_DATA_ID = 41215

OPENML_FALLBACKS = {
    "freMTPL2freq": [
        "https://www.openml.org/data/get_csv/20649148/freMTPL2freq.csv",
    ],
    "freMTPL2sev": [
        "https://www.openml.org/data/get_csv/20649149/freMTPL2sev.csv",
    ],
}

CORE_CATEGORICAL_COLUMNS = ["Area", "VehBrand", "VehGas", "Region"]
CORE_NUMERIC_COLUMNS = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density", "Exposure"]
OPENML_CACHE_DIR = DATA_DIR / "openml_cache"


def _strip_quoted_strings(df: pd.DataFrame) -> pd.DataFrame:
    for column_name in df.columns[df.dtypes == object]:
        df[column_name] = df[column_name].astype(str).str.strip("'")
    return df


def _load_csv_via_requests(urls: Iterable[str], dataset_name: str) -> pd.DataFrame:
    last_error: Exception | None = None
    for url in urls:
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            return pd.read_csv(io.StringIO(response.text))
        except Exception as exc:  # pragma: no cover - network fallback
            last_error = exc
    raise RuntimeError(f"Failed to download {dataset_name} from fallback URLs") from last_error


def _load_openml_or_fallback() -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        OPENML_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        freq_df = fetch_openml(
            data_id=FREQ_DATA_ID,
            as_frame=True,
            data_home=str(OPENML_CACHE_DIR),
        ).data
        sev_df = fetch_openml(
            data_id=SEV_DATA_ID,
            as_frame=True,
            data_home=str(OPENML_CACHE_DIR),
        ).data
    except Exception:
        freq_df = _load_csv_via_requests(OPENML_FALLBACKS["freMTPL2freq"], "freMTPL2freq")
        sev_df = _load_csv_via_requests(OPENML_FALLBACKS["freMTPL2sev"], "freMTPL2sev")
    return freq_df, sev_df


def _create_feature_bands(df: pd.DataFrame) -> pd.DataFrame:
    df["driver_age_band"] = pd.cut(
        df["DrivAge"],
        bins=[17, 25, 35, 50, 65, 100],
        labels=["18-25", "26-35", "36-50", "51-65", "66+"],
        include_lowest=True,
    ).astype(str)
    df["vehicle_age_band"] = pd.cut(
        df["VehAge"],
        bins=[-1, 2, 5, 10, 20, 200],
        labels=["0-2", "3-5", "6-10", "11-20", "21+"],
        include_lowest=True,
    ).astype(str)
    df["bonus_malus_band"] = pd.cut(
        df["BonusMalus"],
        bins=[-1, 75, 100, 125, 150, 1000],
        labels=["<=75", "76-100", "101-125", "126-150", "150+"],
        include_lowest=True,
    ).astype(str)
    df["density_band"] = build_density_bands(df["Density"])
    return df


def build_processed_dataset() -> pd.DataFrame:
    """Load, join, clean, and save the processed freMTPL2 pricing dataset."""
    ensure_project_dirs()
    freq_df, sev_df = _load_openml_or_fallback()

    freq_df = _strip_quoted_strings(freq_df.copy())
    sev_df = _strip_quoted_strings(sev_df.copy())

    freq_df["IDpol"] = freq_df["IDpol"].astype(int)
    sev_df["IDpol"] = sev_df["IDpol"].astype(int)

    sev_agg = (
        sev_df.groupby("IDpol", as_index=False)["ClaimAmount"]
        .sum()
        .rename(columns={"ClaimAmount": "total_claim_amount"})
    )
    df = freq_df.merge(sev_agg, how="left", on="IDpol")
    df["total_claim_amount"] = df["total_claim_amount"].fillna(0.0)

    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    df["Exposure"] = df["Exposure"].clip(lower=1e-6, upper=1.0)
    df.loc[(df["total_claim_amount"] == 0) & (df["ClaimNb"] >= 1), "ClaimNb"] = 0

    df["claim_frequency"] = safe_divide(df["ClaimNb"], df["Exposure"])
    df["pure_premium_raw"] = safe_divide(df["total_claim_amount"], df["Exposure"])

    amount_cap = float(df["total_claim_amount"].quantile(0.99))
    pure_cap = float(df["pure_premium_raw"].quantile(0.99))
    df["total_claim_amount"] = df["total_claim_amount"].clip(upper=amount_cap)
    df["pure_premium"] = df["pure_premium_raw"].clip(upper=pure_cap)

    df = _create_feature_bands(df)

    for column in CORE_CATEGORICAL_COLUMNS:
        df[column] = df[column].astype(str)

    df["portfolio_segment"] = pd.qcut(
        df["pure_premium"].rank(method="first"),
        q=[0.0, 0.7, 0.9, 1.0],
        labels=["Core", "Elevated", "High Risk"],
    ).astype(str)

    saved_path = save_processed_data(df)
    metadata = {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "processed_path": str(saved_path),
        "csv_path": str(PROCESSED_CSV_PATH),
        "amount_cap_99pct": amount_cap,
        "pure_premium_cap_99pct": pure_cap,
        "categorical_columns": CORE_CATEGORICAL_COLUMNS,
        "numeric_columns": CORE_NUMERIC_COLUMNS,
    }
    save_json(metadata, METADATA_JSON_PATH)
    return df


def main() -> None:
    df = build_processed_dataset()
    print(
        "Processed dataset saved with "
        f"{df.shape[0]:,} rows and {df.shape[1]} columns."
    )


if __name__ == "__main__":
    main()
