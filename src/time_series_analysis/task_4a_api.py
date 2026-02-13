"""Task 4A — FastAPI Service

Endpoints:
  POST /predict  — accept vitals, return anomaly flag + risk score + confidence
  GET  /health   — liveness check
"""

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from time_series_analysis.constants import (
    COL_DBP,
    COL_HR,
    COL_MOTION,
    COL_SBP,
    COL_SPO2,
    WINDOW_SIZE_SEC,
)
from time_series_analysis.task_2a_anomaly_detection import load_model, predict_window
from time_series_analysis.task_2b_risk_scoring import compute_risk

_CLF = None
_FEATURE_COLS = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _CLF, _FEATURE_COLS
    try:
        _CLF, _FEATURE_COLS = load_model()
    except FileNotFoundError:
        pass
    yield


app = FastAPI(
    title="Ambulance Vitals Anomaly Detection",
    version="0.2.0",
    lifespan=lifespan,
)


class VitalsInput(BaseModel):
    hr: list[float] | float
    spo2: list[float] | float
    sbp: list[float] | float
    dbp: list[float] | float
    motion: list[float] | float = 0.0


class PredictionOutput(BaseModel):
    anomaly_flag: bool
    risk_score: float
    confidence: float
    alert_status: str


def _to_list(v) -> list:
    return [v] if isinstance(v, (int, float)) else list(v)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _CLF is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(vitals: VitalsInput):
    """Accept vitals and return anomaly flag, risk score, and confidence."""
    data = {
        COL_HR: _to_list(vitals.hr),
        COL_SPO2: _to_list(vitals.spo2),
        COL_SBP: _to_list(vitals.sbp),
        COL_DBP: _to_list(vitals.dbp),
        COL_MOTION: _to_list(vitals.motion),
    }

    # Ensure all columns have same length
    max_len = max(len(v) for v in data.values())
    for key, val in data.items():
        if len(val) == 1 and max_len > 1:
            data[key] = val * max_len

    df = pd.DataFrame(data)

    # Risk scoring (always available)
    risk_result = compute_risk(df)

    # Anomaly detection (requires model & window)
    anomaly_flag = False
    if _CLF is not None and len(df) >= WINDOW_SIZE_SEC:
        pred = predict_window(_CLF, _FEATURE_COLS, df)
        anomaly_flag = bool(pred == 1)
    elif risk_result["risk_score"] >= 70:
        # If model missing, use risk_score >= 70 (higher FP risk)
        anomaly_flag = True

    return PredictionOutput(
        anomaly_flag=anomaly_flag,
        risk_score=risk_result["risk_score"],
        confidence=risk_result["confidence"],
        alert_status=risk_result["alert_status"],
    )
