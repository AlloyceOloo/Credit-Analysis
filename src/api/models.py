# api/models.py
from pydantic import BaseModel
from typing import Dict, List

class ApplicantRequest(BaseModel):
    # keep fields flexible – LGBM uses numeric features only
    age: float
    income: float
    debt: float
    credit_limit: float
    revolving_balance: float
    num_past_due: float
    days_past_due: float
    dti: float
    utilization: float

class ScoreResponse(BaseModel):
    pd: float
    score: float
    shap_top_features: Dict[str, float]

class ScoreResponseBatch(BaseModel):
    results: List[ScoreResponse]
