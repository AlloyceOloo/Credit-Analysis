import joblib
from pathlib import Path
import pandas as pd
import numpy as np

def load_models(models_dir="models"):
    models_dir = Path(models_dir)
    lgbm_artifact = joblib.load(models_dir / "lgbm_model.joblib")
    scorecard_artifact = joblib.load(models_dir / "scorecard.joblib")
    return lgbm_artifact["model"], lgbm_artifact["feature_names"], scorecard_artifact

def predict(df: pd.DataFrame, lgbm_model, lgbm_features, scorecard_artifact):
    """
    Returns dict with 'lgbm_proba' and 'scorecard_score' for each row.
    """
    X = df[lgbm_features].fillna(0)
    y_pred_proba = lgbm_model.predict_proba(X)[:,1]

    # Scorecard prediction
    scorecard_dict = scorecard_artifact["scorecard_dict"]
    numeric_cols = scorecard_artifact["numeric_cols"]
    scores = []
    for _, row in df[numeric_cols].iterrows():
        total = 0
        for feat, bins in scorecard_dict.items():
            value = row[feat]
            for b in bins:
                low, high = b["bin"]
                if low <= value < high:
                    total += b["points"]
                    break
        scores.append(total)
    return pd.DataFrame({
        "lgbm_proba": y_pred_proba,
        "scorecard_score": scores
    })

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--out_csv", default="predictions.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    lgbm_model, lgbm_features, scorecard_artifact = load_models(args.models_dir)
    preds = predict(df, lgbm_model, lgbm_features, scorecard_artifact)
    preds.to_csv(args.out_csv, index=False)
    print(f"Predictions saved to {args.out_csv}")
