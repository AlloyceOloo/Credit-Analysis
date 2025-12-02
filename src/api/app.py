# api/app.py
from fastapi import FastAPI
from api.models import ApplicantRequest, ScoreResponseBatch, ScoreResponse
from api.utils import load_artifacts, predict_single, predict_batch

app = FastAPI(title="CreditX Scoring API", version="1.0")

lgbm_artifact, scorecard, shap_explainer, feature_names = load_artifacts()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/score", response_model=ScoreResponse)
def score_applicant(req: ApplicantRequest):
    return predict_single(req, lgbm_artifact, scorecard, shap_explainer, feature_names)

@app.post("/score_batch", response_model=ScoreResponseBatch)
def score_batch(reqs: list[ApplicantRequest]):
    return predict_batch(reqs, lgbm_artifact, scorecard, shap_explainer, feature_names)
