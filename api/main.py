from __future__ import annotations

import sys
import logging
from time import perf_counter
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from heart_disease_mlop.data_pipeline import FEATURE_COLUMNS  # noqa: E402
from heart_disease_mlop.train_model import load_or_train_model  # noqa: E402

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")
model = load_or_train_model()
logger = logging.getLogger("heart_disease_api")
REQUEST_COUNT = Counter(
    "heart_disease_api_requests_total",
    "HTTP requests served by the heart disease API.",
    ["method", "route", "status"],
)
REQUEST_LATENCY = Histogram(
    "heart_disease_api_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["method", "route"],
)


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    started_at = perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration = perf_counter() - started_at
        route = request.scope.get("route")
        route_name = route.path if route else "unmatched"
        REQUEST_COUNT.labels(request.method, route_name, "500").inc()
        REQUEST_LATENCY.labels(request.method, route_name).observe(duration)
        logger.exception("api_request method=%s route=%s status=500", request.method, route_name)
        raise

    duration = perf_counter() - started_at
    route = request.scope.get("route")
    route_name = route.path if route else "unmatched"
    REQUEST_COUNT.labels(request.method, route_name, str(response.status_code)).inc()
    REQUEST_LATENCY.labels(request.method, route_name).observe(duration)
    logger.info(
        "api_request method=%s route=%s status=%s duration_seconds=%.4f",
        request.method,
        route_name,
        response.status_code,
        duration,
    )
    return response


class PatientRecord(BaseModel):
    age: int = Field(..., ge=0)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: int = Field(..., ge=0)
    chol: int = Field(..., ge=0)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: int = Field(..., ge=0)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0.0)
    slope: int = Field(..., ge=0, le=2)
    ca: int = Field(..., ge=0, le=4)
    thal: int = Field(..., ge=0, le=3)


@app.get("/")
def index():
    return {"message": "Heart disease predictor ready"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(record: PatientRecord):
    payload = record.model_dump()
    df = pd.DataFrame([payload], columns=FEATURE_COLUMNS)
    probabilities = model.predict_proba(df)[0]
    prediction = int(model.predict(df)[0])
    probability = float(probabilities[1])
    confidence = float(max(probabilities))

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "confidence": round(confidence, 4),
    }
