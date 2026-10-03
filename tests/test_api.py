"""
Integration tests for FastAPI REST API endpoints using Starlette/FastAPI TestClient.
"""
import pytest
from fastapi.testclient import TestClient

from src.data_loader import load_raw_data, clean_dataset, get_stratified_splits
from src.pipeline import build_production_pipeline
from src.export_onnx import save_pipeline_joblib
from src.config import MODEL_JOBLIB_PATH
from src.app import app


@pytest.fixture(scope="module", autouse=True)
def setup_model_for_api():
    # Fit and save pipeline model if not present so API can load it
    df_raw = load_raw_data()
    df_clean = clean_dataset(df_raw)
    X_train, y_train, _, _, _, _ = get_stratified_splits(df_clean)

    pipe = build_production_pipeline()
    pipe.fit(X_train, y_train)
    save_pipeline_joblib(pipe, MODEL_JOBLIB_PATH)


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data


def test_metadata_endpoint():
    client = TestClient(app)
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert "disclaimer" in data
    assert "diagnostic" not in data["disclaimer"].lower() or "not a certified medical diagnostic" in data["disclaimer"].lower()


def test_predict_single_endpoint():
    client = TestClient(app)
    payload = {
        "Pregnancies": 2,
        "Glucose": 140.0,
        "BloodPressure": 75.0,
        "SkinThickness": 28.0,
        "Insulin": 120.0,
        "BMI": 31.5,
        "DiabetesPedigreeFunction": 0.42,
        "Age": 45
    }
    response = client.post("/predict?threshold=0.35", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_probability" in data
    assert 0.0 <= data["risk_probability"] <= 1.0
    assert data["risk_class"] in [0, 1]
    assert data["risk_tier"] in ["Low", "Moderate", "High"]
    assert "disclaimer" in data


def test_predict_invalid_schema():
    client = TestClient(app)
    # Invalid: Age is negative, Glucose exceeds valid range
    payload = {
        "Pregnancies": 2,
        "Glucose": 550.0,
        "BloodPressure": 75.0,
        "SkinThickness": 28.0,
        "Insulin": 120.0,
        "BMI": 31.5,
        "DiabetesPedigreeFunction": 0.42,
        "Age": -5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Pydantic validation error


def test_predict_batch_endpoint():
    client = TestClient(app)
    payload = {
        "patients": [
            {
                "Pregnancies": 1,
                "Glucose": 85.0,
                "BloodPressure": 66.0,
                "SkinThickness": 29.0,
                "Insulin": 0.0,
                "BMI": 26.6,
                "DiabetesPedigreeFunction": 0.35,
                "Age": 28
            },
            {
                "Pregnancies": 5,
                "Glucose": 165.0,
                "BloodPressure": 82.0,
                "SkinThickness": 35.0,
                "Insulin": 180.0,
                "BMI": 38.0,
                "DiabetesPedigreeFunction": 0.85,
                "Age": 52
            }
        ]
    }
    response = client.post("/predict/batch?threshold=0.35", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
