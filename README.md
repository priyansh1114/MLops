# Heart Disease MLOps Project

This project implements a complete MLOps workflow for the Heart Disease UCI dataset. It covers data acquisition, EDA, preprocessing, model training, experiment tracking with MLflow, validation tests, Dockerized API deployment, CI/CD automation, and Kubernetes deployment manifests.

## Project structure

- `src/heart_disease_mlop/` – reusable data and training logic
- `api/` – FastAPI application for serving predictions
- `tests/` – unit tests for preprocessing and API behavior
- `k8s/` – Kubernetes deployment files
- `.github/workflows/` – CI pipeline definition
- `artifacts/` – trained model and metadata
- `data/` – downloaded dataset
- `mlruns/` – MLflow experiment metadata

## Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Train the model and create artifacts:

```bash
python -m src.heart_disease_mlop.train_model
```

3. Run the API locally:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

4. Test the prediction endpoint:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 52,
    "sex": 1,
    "cp": 0,
    "trestbps": 125,
    "chol": 212,
    "fbs": 0,
    "restecg": 1,
    "thalach": 168,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 2,
    "thal": 2
  }'
```

## Dataset source

The project downloads the Cleveland Heart Disease dataset from the UCI archive if it is not already present locally.

## Model and experiments

- Model training uses a preprocessing pipeline and compares Logistic Regression and Random Forest classifiers.
- The best model is selected by ROC-AUC.
- MLflow logs parameters and metrics to `mlruns/`.

## Docker

```bash
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api
```

## Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
```

## CI/CD

GitHub Actions runs lint/test steps and model training validation on pushes and pull requests.

## Report

See `report.md` for the assignment-style project summary and architecture overview.
