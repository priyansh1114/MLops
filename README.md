# Heart Disease MLOps

End-to-end classifier for the UCI Cleveland Heart Disease dataset. The repository includes data acquisition and EDA, preprocessing, tuned model training, MLflow tracking, a JSON inference command, a FastAPI service, Prometheus metrics, tests, CI, Docker packaging, and Kubernetes manifests.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The dataset is stored in `data/heart.csv`. If it is absent, `load_dataset()` downloads the Cleveland source file from the [UCI archive](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data). The original 0-4 target is mapped to a binary task: 0 means no disease and 1-4 means disease.

## EDA

```powershell
python -m src.heart_disease_mlop.eda
```

This creates feature histograms, a correlation heatmap, a target class balance chart, and `artifacts/eda/summary.json`. The summary records row count, missing values, and class counts. Missing feature values are imputed inside the training pipeline, avoiding whole-dataset imputation before the train/test split.

## Train and track experiments

```powershell
python -m src.heart_disease_mlop.train_model
```

The training command makes a stratified 80/20 split, performs five-fold stratified grid search for Logistic Regression and Random Forest, and selects by mean cross-validated ROC-AUC. It reports accuracy, precision, recall, and ROC-AUC for cross-validation and the holdout set, and logs parameters, metrics, model files, grid-search results, and ROC/confusion-matrix figures to local MLflow tracking at `mlruns/`.

To browse runs locally:

```powershell
$env:MLFLOW_ALLOW_FILE_STORE = "true"
mlflow ui --backend-store-uri .\mlruns --host 127.0.0.1 --port 5000
```

The selected complete preprocessing/model pipeline is saved at `artifacts/best_model.joblib`; per-model evaluation files and `artifacts/training_summary.json` are also produced.

## Inference and API

The command-line inference interface accepts one JSON object with all 13 features:

```powershell
python -m src.heart_disease_mlop.predict '{"age":52,"sex":1,"cp":0,"trestbps":125,"chol":212,"fbs":0,"restecg":1,"thalach":168,"exang":0,"oldpeak":1.0,"slope":2,"ca":2,"thal":2}'
```

Run the API:

```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

- `GET /health` reports service/model readiness.
- `POST /predict` accepts the 13 patient features and returns `prediction` (0 or 1), `probability` (probability of class 1), and `confidence` (probability of the predicted class).
- `GET /metrics` exposes Prometheus request counters and request-duration histograms.

Requests are logged with method, route, status, and duration; patient feature values are not included in logs.

## Tests and lint

```powershell
ruff check --select E4,E7,E9,F src api tests
pytest -q
```

GitHub Actions runs lint, tests, model training, and EDA on pushes and pull requests to `main` or `master`. It uploads model, MLflow, and plot outputs as workflow artifacts.

## Docker

With Docker Desktop running:

```powershell
docker build -t heart-disease-api .
docker run --name heart-disease-api-local -p 8000:8000 heart-disease-api
```

Open `http://localhost:8000/docs` for interactive API documentation. The container exposes port 8000 and starts the FastAPI app.
Verify the deployed container with `http://localhost:8000/health`, `POST /predict`, and `http://localhost:8000/metrics`. To stop and remove the named container, run `docker rm -f heart-disease-api-local`.

## Kubernetes and monitoring

The manifest in `k8s/deployment.yaml` creates two API replicas and a `LoadBalancer` service, with readiness/liveness probes. Make the image available to the chosen cluster, then:

```powershell
.\scripts\load_docker_desktop_image.ps1
kubectl apply -f k8s/deployment.yaml
kubectl rollout status deployment/heart-disease-api --timeout=180s
kubectl get deployment,pods,svc -o wide
kubectl port-forward service/heart-disease-api 8080:80
```

Open `http://localhost:8080/docs` while port forwarding is active to verify the Kubernetes-backed API. Docker Desktop Kubernetes uses a separate containerd image store in this setup; the PowerShell helper imports the local image into that runtime, and `imagePullPolicy: Never` prevents an unintended pull of a nonexistent public image. For Minikube, use `minikube image load heart-disease-api:latest` instead. `monitoring/prometheus.yml` contains a Prometheus scrape job for the API `/metrics` endpoint; update its target if Prometheus is outside the Kubernetes cluster.

## Assignment report and verification status

See [report.pdf](report.pdf) for the 10-page submission report and [report.html](report.html) for its printable source. [report.md](report.md) contains the detailed technical report. EDA outputs are under `artifacts/eda/`, and live Docker API screenshots are in `screenshots/`.

Local validation passed: all five tests and Ruff checks passed, and the Docker image and two-replica Docker Desktop Kubernetes deployment returned successful health, prediction, and metrics responses. The latest GitHub Actions run [#7 passed](https://github.com/priyansh1114/MLops/actions/runs/36999886352) lint, tests, training, EDA, and artifact upload; the artifact is available [here](https://github.com/priyansh1114/MLops/actions/runs/36999886352/artifacts/11223470815). The Docker container is named `heart-disease-api-local` on port 8000; the Kubernetes service was verified with `kubectl port-forward` on port 8080. All screenshots are embedded in `report.pdf` and available under `screenshots/`. No public API URL is claimed.

This educational model is not a medical device and must not be used for clinical decisions. The Cleveland subset is small; its metrics do not establish safety, generalization, calibration, or clinical utility.