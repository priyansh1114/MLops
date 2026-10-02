from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metrics_endpoint_exposes_request_metrics():
    client.get("/health")
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "heart_disease_api_requests_total" in response.text
    assert "heart_disease_api_request_duration_seconds" in response.text


def test_predict_endpoint():
    payload = {
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
        "thal": 2,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "confidence" in data
    assert data["prediction"] in {0, 1}
    assert 0.0 <= data["probability"] <= 1.0
