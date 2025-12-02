import pandas as pd
import numpy as np
from src.features.fe_build import build_features

def test_basic_feature_shapes():
    df = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "age": [25, 40, 60],
        "income": [30000, 70000, 120000],
        "debt": [2000, 5000, 10000],
        "credit_limit": [10000, 20000, 30000],
        "revolving_balance": [2000, 4000, 6000],
        "num_past_due": [0, 1, 0],
        "days_past_due": [0, 20, 0],
        "dti": [0.1, 0.2, 0.3],
        "utilization": [0.2, 0.3, 0.4],
        "target": [0, 1, 0],
    })

    out = build_features(df)

    expected_cols = {
        "age","income","income_log","debt","credit_limit",
        "revolving_balance","dti_calc","util_calc",
        "debt_income_ratio","revol_to_debt",
        "num_past_due","days_past_due",
        "ever_past_due","recent_dpd","high_util_and_dpd",
        "risk_score","age_band","income_bucket",
        "customer_id","target",
    }

    assert set(out.columns) == expected_cols
    assert len(out) == 3
    assert out["income_log"].iloc[0] == np.log1p(30000)
