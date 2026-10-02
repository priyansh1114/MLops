from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

try:
    from .data_pipeline import MODEL_DIR, ROOT, FEATURE_COLUMNS, build_preprocessor, load_dataset
except ImportError:  # pragma: no cover
    from heart_disease_mlop.data_pipeline import MODEL_DIR, ROOT, FEATURE_COLUMNS, build_preprocessor, load_dataset


def evaluate_model(model_pipeline, X_test, y_test):
    probabilities = model_pipeline.predict_proba(X_test)[:, 1]
    predictions = model_pipeline.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def train_best_model() -> Pipeline:
    df = load_dataset()
    X = df[FEATURE_COLUMNS]
    y = df["target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
        ),
    }

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = (ROOT / "mlruns").as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("heart_disease_prediction")

    best_name = ""
    best_results = {}
    best_pipeline = None
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="heart-disease-training") as run:
        for name, model in models.items():
            with mlflow.start_run(run_name=f"model_{name}", nested=True):
                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", build_preprocessor()),
                        ("model", model),
                    ]
                )

                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cross_val_scores = cross_val_score(
                    pipeline,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring="roc_auc",
                )

                pipeline.fit(X_train, y_train)
                metrics = evaluate_model(pipeline, X_test, y_test)

                mlflow.log_params({
                    "model_name": name,
                    "test_size": 0.2,
                    "random_state": 42,
                    "cv_folds": 5,
                })
                mlflow.log_metrics({
                    "cv_roc_auc_mean": float(cross_val_scores.mean()),
                    "cv_roc_auc_std": float(cross_val_scores.std()),
                    **{f"{key}": float(value) for key, value in metrics.items()},
                })

                model_artifact = MODEL_DIR / f"{name}_model.joblib"
                joblib.dump(pipeline, model_artifact)
                mlflow.log_artifact(str(model_artifact), artifact_path=f"models/{name}")

                result = {"model_name": name, **metrics, "cv_roc_auc_mean": float(cross_val_scores.mean())}
                if not best_name or result["roc_auc"] > best_results.get("roc_auc", -1):
                    best_name = name
                    best_results = result
                    best_pipeline = pipeline

        mlflow.log_param("best_model", best_name)
        mlflow.log_metric("best_roc_auc", float(best_results["roc_auc"]))

    if best_pipeline is None:
        raise RuntimeError("No valid model trained.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(best_pipeline, model_path)

    summary = {
        "best_model": best_name,
        "metrics": best_results,
        "model_path": str(model_path),
    }
    (MODEL_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return best_pipeline


def load_or_train_model(model_path: Path | str | None = None):
    path = Path(model_path) if model_path else MODEL_DIR / "best_model.joblib"
    if path.exists():
        return joblib.load(path)

    return train_best_model()


if __name__ == "__main__":
    train_best_model()
