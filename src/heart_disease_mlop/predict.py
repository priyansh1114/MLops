from __future__ import annotations

import argparse
import json

import pandas as pd

from .data_pipeline import FEATURE_COLUMNS
from .train_model import load_or_train_model


def predict_record(record: dict[str, int | float]) -> dict[str, int | float]:
    """Return class prediction, disease probability, and predicted-class confidence."""
    model = load_or_train_model()
    frame = pd.DataFrame([{column: record[column] for column in FEATURE_COLUMNS}])
    probabilities = model.predict_proba(frame)[0]
    prediction = int(model.predict(frame)[0])
    return {
        "prediction": prediction,
        "probability": float(probabilities[1]),
        "confidence": float(max(probabilities)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict heart disease from a JSON patient record.")
    parser.add_argument("record", help="JSON object containing the 13 model features")
    arguments = parser.parse_args()
    print(json.dumps(predict_record(json.loads(arguments.record)), indent=2))


if __name__ == "__main__":
    main()