from __future__ import annotations
from typing import List
import pandas as pd
import numpy as np


NUMERIC_FEATURES = [
    "age", "income", "debt", "credit_limit", "revolving_balance",
    "num_past_due", "days_past_due", "dti", "utilization"
]


def _age_band(age: pd.Series) -> pd.Series:
    bins = [0, 25, 35, 45, 55, 65, 200]
    labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
    return pd.cut(age, bins=bins, labels=labels, right=True)


def _income_bucket(income: pd.Series) -> pd.Series:
    bins = [0, 20000, 40000, 60000, 100000, 1e9]
    labels = ["0-20k", "20-40k", "40-60k", "60-100k", "100k+"]
    return pd.cut(income, bins=bins, labels=labels, right=True)


def build_features(df: pd.DataFrame,
                   keep_cols: List[str] | None = None) -> pd.DataFrame:

    df = df.copy()

    # Ensure required numeric columns exist
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0

    # Base engineered fields
    df["dti_calc"] = (df["debt"] / (df["income"] + 1e-9)).replace([np.inf, -np.inf], 0).fillna(0)
    df["util_calc"] = (df["revolving_balance"] / (df["credit_limit"] + 1e-9)).replace([np.inf, -np.inf], 0).fillna(0)

    # Additional ratios
    df["debt_income_ratio"] = df["debt"] / (df["income"] + 1e-9)
    df["revol_to_debt"] = df["revolving_balance"] / (df["debt"] + 1e-9)

    # Binary delinquency indicators
    df["ever_past_due"] = (df["num_past_due"] > 0).astype(int)
    df["recent_dpd"] = ((df["days_past_due"] <= 30) & (df["ever_past_due"] == 1)).astype(int)

    # Interaction signal
    df["high_util_and_dpd"] = ((df["util_calc"] > 0.8) & (df["ever_past_due"] == 1)).astype(int)

    # Categorical buckets
    df["age_band"] = _age_band(df["age"])
    df["income_bucket"] = _income_bucket(df["income"])

    # Log transform
    df["income_log"] = np.log1p(df["income"])

    # Simple composite risk indicator
    df["risk_score"] = (
        0.5 * df["dti_calc"] +
        0.3 * df["util_calc"] +
        0.2 * df["ever_past_due"]
    )

    feature_cols = [
        "age", "income", "income_log", "debt", "credit_limit",
        "revolving_balance", "dti_calc", "util_calc",
        "debt_income_ratio", "revol_to_debt",
        "num_past_due", "days_past_due",
        "ever_past_due", "recent_dpd", "high_util_and_dpd",
        "risk_score", "age_band", "income_bucket"
    ]

    keep_cols = keep_cols or ["customer_id", "target"]
    keep_cols = [c for c in keep_cols if c in df.columns]

    out = df[feature_cols + keep_cols].copy()

    # Make categorical safe for Parquet
    out["age_band"] = out["age_band"].astype(str)
    out["income_bucket"] = out["income_bucket"].astype(str)

    return out


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/processed/customers.parquet")
    parser.add_argument("--out", default="data/features/features.parquet")
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"{src} does not exist. Run ingestion first.")

    df = pd.read_parquet(src)
    df_out = build_features(df)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(args.out, index=False)

    print(f"Wrote {len(df_out)} features → {args.out}")
