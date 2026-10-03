"""
CO6: Production REST Service using FastAPI for Diabetes Risk Screening.
"""
from typing import List, Dict, Any, Optional
import os
import pathlib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

STATIC_DIR = pathlib.Path(__file__).parent / "static"

from src.config import (
    MODEL_JOBLIB_PATH,
    CALIBRATED_MODEL_PATH,
    CLINICAL_DISCLAIMER,
    FEATURE_COLUMNS
)
from src.export_onnx import load_pipeline_joblib


app = FastAPI(
    title="Diabetes Risk Prediction Screening Service",
    description=(
        "Production-style ML inference endpoint for screening diabetes risk onset. "
        "Strictly for clinical triage support; NOT a diagnostic device."
    ),
    version="1.0.0"
)

# Serve the frontend at /
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_frontend():
    """Serve the DiaRisk AI frontend."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse("<h2>Frontend not found. Run the full pipeline first.</h2>", status_code=404)

# Mount static assets (CSS/JS/images) under /static
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Load pipeline at startup if available
pipeline = None
if os.path.exists(CALIBRATED_MODEL_PATH):
    pipeline = load_pipeline_joblib(CALIBRATED_MODEL_PATH)
elif os.path.exists(MODEL_JOBLIB_PATH):
    pipeline = load_pipeline_joblib(MODEL_JOBLIB_PATH)


class PatientProfile(BaseModel):
    Pregnancies: int = Field(..., ge=0, le=25, description="Number of pregnancies")
    Glucose: float = Field(..., ge=0, le=300, description="Plasma glucose concentration (0 represents unmeasured/missing)")
    BloodPressure: float = Field(..., ge=0, le=200, description="Diastolic blood pressure mm Hg (0 represents unmeasured/missing)")
    SkinThickness: float = Field(..., ge=0, le=120, description="Triceps skin fold thickness mm (0 represents unmeasured/missing)")
    Insulin: float = Field(..., ge=0, le=900, description="2-Hour serum insulin mu U/ml (0 represents unmeasured/missing)")
    BMI: float = Field(..., ge=0, le=80, description="Body Mass Index kg/m^2 (0 represents unmeasured/missing)")
    DiabetesPedigreeFunction: float = Field(..., ge=0.0, le=3.0, description="Diabetes pedigree genetic score")
    Age: int = Field(..., ge=1, le=120, description="Age in years")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Pregnancies": 2,
                "Glucose": 138.0,
                "BloodPressure": 74.0,
                "SkinThickness": 26.0,
                "Insulin": 140.0,
                "BMI": 32.4,
                "DiabetesPedigreeFunction": 0.45,
                "Age": 38
            }
        }
    }


class PredictionResult(BaseModel):
    risk_probability: float
    risk_class: int
    risk_tier: str
    operational_threshold: float
    disclaimer: str
    clinical_recommendation: str


class BatchPredictionRequest(BaseModel):
    patients: List[PatientProfile]


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health and readiness check endpoint."""
    return {
        "status": "healthy",
        "service": "Diabetes Risk Screening Endpoint",
        "model_loaded": pipeline is not None
    }


@app.get("/metadata", status_code=status.HTTP_200_OK)
def get_metadata():
    """Model governance metadata, schema definitions, and clinical caution."""
    return {
        "service_name": "Diabetes Risk Prediction",
        "model_architecture": "Ensemble Pipeline with Missingness Indicators & Platt Calibration",
        "features": FEATURE_COLUMNS,
        "primary_clinical_metric": "Recall / PR-AUC (Priority on minimizing False Negatives)",
        "disclaimer": CLINICAL_DISCLAIMER
    }


@app.post("/predict", response_model=PredictionResult)
def predict_single(patient: PatientProfile, threshold: Optional[float] = 0.35):
    """
    Predict diabetes risk onset probability for a single patient record.
    """
    global pipeline
    if pipeline is None:
        if os.path.exists(CALIBRATED_MODEL_PATH):
            pipeline = load_pipeline_joblib(CALIBRATED_MODEL_PATH)
        elif os.path.exists(MODEL_JOBLIB_PATH):
            pipeline = load_pipeline_joblib(MODEL_JOBLIB_PATH)
        else:
            raise HTTPException(status_code=503, detail="Trained model artifact not found.")

    patient_dict = patient.model_dump() if hasattr(patient, "model_dump") else patient.dict()
    df_in = pd.DataFrame([patient_dict])
    prob = float(pipeline.predict_proba(df_in)[0, 1])
    risk_class = int(prob >= threshold)

    if prob < 0.30:
        tier = "Low"
        rec = "Routine wellness monitoring; maintain healthy dietary lifestyle."
    elif prob < 0.60:
        tier = "Moderate"
        rec = "Recommended for lifestyle intervention and secondary prediabetes screening."
    else:
        tier = "High"
        rec = "Prioritized clinical follow-up: formal HbA1c and Oral Glucose Tolerance Test recommended."

    return PredictionResult(
        risk_probability=round(prob, 4),
        risk_class=risk_class,
        risk_tier=tier,
        operational_threshold=round(threshold, 3),
        disclaimer=CLINICAL_DISCLAIMER,
        clinical_recommendation=rec
    )


@app.post("/predict/batch")
def predict_batch(batch: BatchPredictionRequest, threshold: Optional[float] = 0.35):
    """
    Batch risk prediction endpoint for multi-patient screening workflows.
    """
    global pipeline
    if pipeline is None:
        if os.path.exists(CALIBRATED_MODEL_PATH):
            pipeline = load_pipeline_joblib(CALIBRATED_MODEL_PATH)
        elif os.path.exists(MODEL_JOBLIB_PATH):
            pipeline = load_pipeline_joblib(MODEL_JOBLIB_PATH)
        else:
            raise HTTPException(status_code=503, detail="Trained model artifact not found.")

    patients_list = [p.model_dump() if hasattr(p, "model_dump") else p.dict() for p in batch.patients]
    df_in = pd.DataFrame(patients_list)
    probs = pipeline.predict_proba(df_in)[:, 1]

    results = []
    for prob in probs:
        p_val = float(prob)
        r_class = int(p_val >= threshold)
        tier = "Low" if p_val < 0.30 else ("Moderate" if p_val < 0.60 else "High")
        results.append({
            "risk_probability": round(p_val, 4),
            "risk_class": r_class,
            "risk_tier": tier
        })

    return {
        "count": len(results),
        "operational_threshold": threshold,
        "predictions": results,
        "disclaimer": CLINICAL_DISCLAIMER
    }
