# api/utils.py
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import shap
from typing import List, Dict, Any

MODELS_DIR = Path("models")
ARTIFACTS_DIR = Path("artifacts")

def load_artifacts():
    # load lgbm artifact saved as dict {'model': lgbm, 'feature_names': [...]}
    lgbm_artifact = joblib.load(MODELS_DIR / "lgbm_model.joblib")
    # scorecard (dict with binner, clf, numeric_cols, pd_to_score)
    scorecard = joblib.load(MODELS_DIR / "scorecard.joblib")
    feature_names = lgbm_artifact.get("feature_names", [])
    # SHAP explainer
    lgbm_model = lgbm_artifact["model"]
    explainer = shap.TreeExplainer(lgbm_model)
    return lgbm_artifact, scorecard, explainer, feature_names

def _build_input_df_from_request(req, feature_names: List[str]) -> pd.DataFrame:
    # req may be Pydantic model
    d = {}
    for f in feature_names:
        # try to map names directly from request attributes if available
        if hasattr(req, f):
            d[f] = getattr(req, f)
        else:
            # fallback: try a few mapping options (dti/utilization exist)
            d[f] = getattr(req, f, 0)
    # if any required feature missing, DataFrame will include it with 0
    return pd.DataFrame([d])[feature_names].fillna(0)

def predict_single(req, lgbm_artifact, scorecard, shap_explainer, feature_names):
    # prepare df
    df = _build_input_df_from_request(req, feature_names)
    lgbm = lgbm_artifact["model"]
    pd_val = float(lgbm.predict_proba(df)[0][1])

    # Scorecard scoring on numeric features
    # construct DataFrame with numeric columns expected by scorecard
    sc_numeric_cols = scorecard["numeric_cols"]
    sc_df = pd.DataFrame([ {c: df.get(c, 0).iloc[0] for c in sc_numeric_cols} ])[sc_numeric_cols]
    sc_res = scorecard_predict(scorecard, sc_df)
    score_val = float(sc_res["score"][0])
    # SHAP local explanation top features
    shap_vals = shap_explainer(df)
    # shap_vals.values shape: (n_samples, n_features)
    per_feat = dict(zip(feature_names, shap_vals.values[0].tolist()))
    # take top 5 by absolute contribution
    top5 = dict(sorted(per_feat.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5])

    return {"pd": pd_val, "score": score_val, "shap_top_features": top5}

def predict_batch(reqs, lgbm_artifact, scorecard, shap_explainer, feature_names):
    results = []
    for r in reqs:
        results.append(predict_single(r, lgbm_artifact, scorecard, shap_explainer, feature_names))
    return {"results": results}

def scorecard_predict(scorecard: Dict[str, Any], X_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Wrapper that uses scorecard dict and returns dict with numpy arrays 'pd' and 'score'.
    """
    X = X_df[scorecard["numeric_cols"]].fillna(0).values
    X_binned = scorecard["binner"].transform(X)
    pd_preds = scorecard["clf"].predict_proba(X_binned)[:, 1]
    scores = scorecard["pd_to_score"](pd_preds)
    return {"pd": pd_preds, "score": scores}
