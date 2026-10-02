from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
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
        "logistic_regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", random_state=42),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 6],
                "model__min_samples_leaf": [1, 2],
            },
        ),
    }

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = (ROOT / "mlruns").as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("heart_disease_prediction")

    best_name = ""
    best_results = {}
    best_pipeline = None
    model_results = []
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="heart-disease-training"):
        for name, (model, parameter_grid) in models.items():
            with mlflow.start_run(run_name=f"model_{name}", nested=True):
                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", build_preprocessor()),
                        ("model", model),
                    ]
                )

                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                search = GridSearchCV(
                    pipeline,
                    parameter_grid,
                    cv=cv,
                    scoring={
                        "accuracy": "accuracy",
                        "precision": "precision",
                        "recall": "recall",
                        "roc_auc": "roc_auc",
                    },
                    refit="roc_auc",
                    n_jobs=1,
                    return_train_score=False,
                )
                search.fit(X_train, y_train)
                best_pipeline_for_model = search.best_estimator_
                metrics = evaluate_model(best_pipeline_for_model, X_test, y_test)
                best_index = search.best_index_
                cv_metrics = {
                    metric: {
                        "mean": float(search.cv_results_[f"mean_test_{metric}"][best_index]),
                        "std": float(search.cv_results_[f"std_test_{metric}"][best_index]),
                    }
                    for metric in ("accuracy", "precision", "recall", "roc_auc")
                }

                mlflow.log_params({
                    "model_name": name,
                    "test_size": 0.2,
                    "random_state": 42,
                    "cv_folds": 5,
                    "selection_metric": "mean_cv_roc_auc",
                    **{f"best_{key}": value for key, value in search.best_params_.items()},
                })
                mlflow.log_metrics({
                    **{
                        f"cv_{metric}_{stat}": value
                        for metric, statistics in cv_metrics.items()
                        for stat, value in statistics.items()
                    },
                    **{f"{key}": float(value) for key, value in metrics.items()},
                })

                model_artifact = MODEL_DIR / f"{name}_model.joblib"
                joblib.dump(best_pipeline_for_model, model_artifact)
                mlflow.log_artifact(str(model_artifact), artifact_path=f"models/{name}")

                evaluation_dir = MODEL_DIR / "evaluation" / name
                evaluation_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(search.cv_results_).to_csv(
                    evaluation_dir / "grid_search_results.csv", index=False
                )
                figure, axes = plt.subplots(1, 2, figsize=(11, 4))
                ConfusionMatrixDisplay.from_estimator(
                    best_pipeline_for_model, X_test, y_test, ax=axes[0], colorbar=False
                )
                RocCurveDisplay.from_estimator(
                    best_pipeline_for_model, X_test, y_test, ax=axes[1]
                )
                figure.tight_layout()
                figure.savefig(evaluation_dir / "holdout_diagnostics.png", dpi=160)
                plt.close(figure)
                mlflow.log_artifacts(str(evaluation_dir), artifact_path=f"evaluation/{name}")

                result = {
                    "model_name": name,
                    "best_params": search.best_params_,
                    "cv_metrics": cv_metrics,
                    **metrics,
                }
                model_results.append(result)
                if not best_name or cv_metrics["roc_auc"]["mean"] > best_results["cv_metrics"]["roc_auc"]["mean"]:
                    best_name = name
                    best_results = result
                    best_pipeline = best_pipeline_for_model

        mlflow.log_param("best_model", best_name)
        mlflow.log_metric("best_cv_roc_auc", best_results["cv_metrics"]["roc_auc"]["mean"])

    if best_pipeline is None:
        raise RuntimeError("No valid model trained.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(best_pipeline, model_path)

    summary = {
        "best_model": best_name,
        "metrics": best_results,
        "model_comparison": model_results,
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
