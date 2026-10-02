from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from heart_disease_mlop.data_pipeline import FEATURE_COLUMNS
from heart_disease_mlop.train_model import load_or_train_model

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")
model = load_or_train_model()


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


@app.post("/predict")
def predict(record: PatientRecord):
    payload = record.model_dump()
    df = pd.DataFrame([payload], columns=FEATURE_COLUMNS)
    probabilities = model.predict_proba(df)[0]
    prediction = int(model.predict(df)[0])
    probability = float(max(probabilities))

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "confidence": round(probability, 4),
    }
