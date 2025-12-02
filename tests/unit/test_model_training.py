# tests/unit/test_model_training.py
import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from scripts.generate_synthetic_data import generate
from src.features.fe_build import build_features
from src.models.train import train

def test_smoke_training(tmp_path):
    # make synthetic, ingest-like output
    df = generate(n=500, seed=123)
    # build features
    feats = build_features(df)
    feat_path = tmp_path / "features.parquet"
    feats.to_parquet(feat_path, index=False)

    out_models = tmp_path / "models"
    out_artifacts = tmp_path / "artifacts"
    out_models.mkdir(exist_ok=True)
    out_artifacts.mkdir(exist_ok=True)

    res = train(str(feat_path), out_dir=str(out_models), artifacts_dir=str(out_artifacts))
    # Check model files exist
    assert Path(res["lgbm_path"]).exists()
    assert Path(res["scorecard_path"]).exists()

    # quick AUC smoke: load artifacts file
    metrics = res["metrics"]
    assert metrics["auc"] >= 0.5  # weak smoke check
