import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier
from lightgbm import early_stopping, log_evaluation

from src.models.scorecard import fit_scorecard


def train(features_path: str,
          out_dir: str = "models",
          artifacts_dir: str = "artifacts",
          random_state: int = 42):

    features_path = Path(features_path)
    if not features_path.exists():
        raise FileNotFoundError(f"{features_path} not found. Build features first.")

    df = pd.read_parquet(features_path)
    if "target" not in df.columns:
        raise ValueError("features must include 'target' column")

    # ----------------------
    # Numeric feature set
    # ----------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "target"]

    X = df[numeric_cols].fillna(0)
    y = df["target"].astype(int)

    # ----------------------
    # Train/val split
    # ----------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2,
        random_state=random_state,
        stratify=y
    )

    callbacks = [
        early_stopping(stopping_rounds=50),
        log_evaluation(period=0)
    ]

    # ----------------------
    # Train LightGBM
    # ----------------------
    lgbm = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        n_jobs=-1
    )

    try:
        lgbm.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks
        )
    except TypeError:
        # Older LightGBM fallback
        lgbm.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)]
        )

    # ----------------------
    # Validation metrics
    # ----------------------
    y_val_proba = lgbm.predict_proba(X_val)[:, 1]
    auc = float(roc_auc_score(y_val, y_val_proba))
    gini = 2 * auc - 1

    # ----------------------
    # ROC Plot
    # ----------------------
    fpr, tpr, _ = roc_curve(y_val, y_val_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(artifacts_dir) / "roc.png")
    plt.close()

    metrics = {
        "auc": auc,
        "gini": gini,
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0])
    }

    # ----------------------
    # Save LGBM model
    # ----------------------
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lgbm_path = out_dir / "lgbm_model.joblib"
    joblib.dump({"model": lgbm, "feature_names": numeric_cols}, lgbm_path)

    # ----------------------
    # Fit WOE Logistic Scorecard
    # ----------------------
    print("Fitting scorecard model...")
    scorecard_dict, discretizer = fit_scorecard(X, y)

    scorecard_path = out_dir / "scorecard.joblib"
    joblib.dump({
        "scorecard_dict": scorecard_dict,
        "discretizer": discretizer,
        "numeric_cols": numeric_cols
    }, scorecard_path)

    # ----------------------
    # Save metadata
    # ----------------------
    with open(Path(artifacts_dir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(Path(artifacts_dir) / "model_info.json", "w") as f:
        json.dump({
            "lgbm_path": str(lgbm_path),
            "scorecard_path": str(scorecard_path),
            "feature_names": numeric_cols
        }, f, indent=2)

    print("\nTraining completed.")
    print("AUC:", auc)
    return {
        "lgbm_path": str(lgbm_path),
        "scorecard_path": str(scorecard_path),
        "metrics": metrics
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/features/features.parquet")
    parser.add_argument("--out_dir", default="models")
    parser.add_argument("--artifacts_dir", default="artifacts")
    args = parser.parse_args()
    train(args.features, args.out_dir, args.artifacts_dir)
