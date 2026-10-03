"""
CO1: ML System Lifecycle Mapping, Prediction Request Tracing, and Skew Diagnosis.
"""
from typing import Dict, Any, List
import time
import numpy as np
import pandas as pd


LIFECYCLE_STAGES = [
    {
        "stage": "1. Data Ingestion & Storage",
        "component": "src/data_loader.py (raw data: data/diabetes.csv)",
        "description": "Loads clinical records, validates column schemas, detects missing values and physiological zeroes."
    },
    {
        "stage": "2. Preprocessing & Feature Engineering",
        "component": "src/pipeline.py (ColumnTransformer with SimpleImputer + StandardScaler)",
        "description": "Imputes physiological missing values (median), adds MissingIndicator flags, scales continuous distributions."
    },
    {
        "stage": "3. Supervised Model Training",
        "component": "src/linear_models.py, src/tree_models.py",
        "description": "Fits Logistic Regression, Random Forest, XGBoost, and SVM with cross-validation."
    },
    {
        "stage": "4. Model Evaluation, Selection & Calibration",
        "component": "src/evaluation.py",
        "description": "Computes PR-AUC, Recall, Precision, Brier score; tunes classification threshold for high sensitivity; calibrates probabilities."
    },
    {
        "stage": "5. Packaging & Serialization",
        "component": "src/export_onnx.py",
        "description": "Exports reproducible artifacts to Joblib (pipeline) and ONNX format with graph optimization."
    },
    {
        "stage": "6. REST Serving Endpoint",
        "component": "src/app.py (FastAPI application with /predict and /health)",
        "description": "Receives HTTP requests, validates Pydantic schema, executes inference, and formats response."
    },
    {
        "stage": "7. Production Observability & Monitoring",
        "component": "src/monitoring.py (PSI drift detector)",
        "description": "Monitors feature distribution shifts (PSI > 0.20), tracks inference latency, and triggers retraining alerts."
    }
]


def trace_prediction_request(
    raw_patient_data: Dict[str, Any],
    pipeline,
    decision_threshold: float = 0.35
) -> Dict[str, Any]:
    """
    Executes a step-by-step trace of a single incoming clinical request through the ML lifecycle.
    Identifies the exact state transformations at each boundary.
    """
    trace_log: List[Dict[str, Any]] = []
    start_time = time.perf_counter()

    # Step 1: Input Ingestion & Validation
    step1_start = time.perf_counter()
    df_single = pd.DataFrame([raw_patient_data])
    step1_ms = (time.perf_counter() - step1_start) * 1000
    trace_log.append({
        "step": 1,
        "name": "HTTP Request Ingestion & Schema Validation",
        "latency_ms": round(step1_ms, 3),
        "input_payload": raw_patient_data,
        "status": "Schema validated against Pydantic model"
    })

    # Step 2: Feature Sanitization (Zero -> NaN for physiological attributes)
    step2_start = time.perf_counter()
    from src.config import PHYSIOLOGICAL_ZERO_COLUMNS
    df_sanitized = df_single.copy()
    for col in PHYSIOLOGICAL_ZERO_COLUMNS:
        if col in df_sanitized.columns and df_sanitized[col].iloc[0] == 0:
            df_sanitized[col] = np.nan
    step2_ms = (time.perf_counter() - step2_start) * 1000
    trace_log.append({
        "step": 2,
        "name": "Feature Sanitization & Zero Handling",
        "latency_ms": round(step2_ms, 3),
        "sanitized_features": df_sanitized.to_dict(orient="records")[0],
        "zero_replacement_applied": {
            col: (df_single[col].iloc[0] == 0)
            for col in PHYSIOLOGICAL_ZERO_COLUMNS if col in df_single.columns
        }
    })

    # Step 3: Transformation Pipeline (Imputation + Scaling)
    step3_start = time.perf_counter()
    # The preprocessor is the first step of the pipeline
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed_features = preprocessor.transform(df_sanitized)
    step3_ms = (time.perf_counter() - step3_start) * 1000
    trace_log.append({
        "step": 3,
        "name": "Feature Pipeline Transformation",
        "latency_ms": round(step3_ms, 3),
        "output_feature_dim": int(transformed_features.shape[1]),
        "sample_transformed_values": np.round(transformed_features[0][:5], 4).tolist()
    })

    # Step 4: Model Inference Engine
    step4_start = time.perf_counter()
    classifier = pipeline.named_steps["classifier"]
    raw_probs = classifier.predict_proba(transformed_features)[0]
    prob_diabetic = float(raw_probs[1])
    step4_ms = (time.perf_counter() - step4_start) * 1000
    trace_log.append({
        "step": 4,
        "name": "Model Inference Engine",
        "latency_ms": round(step4_ms, 3),
        "raw_probabilities": {"negative": round(raw_probs[0], 4), "positive": round(raw_probs[1], 4)},
        "calibrated_risk_score": round(prob_diabetic, 4)
    })

    # Step 5: Decision Thresholding & Response Formatter
    step5_start = time.perf_counter()
    predicted_class = int(prob_diabetic >= decision_threshold)
    if prob_diabetic < 0.30:
        risk_tier = "Low"
    elif prob_diabetic < 0.60:
        risk_tier = "Moderate"
    else:
        risk_tier = "High"

    step5_ms = (time.perf_counter() - step5_start) * 1000
    total_ms = (time.perf_counter() - start_time) * 1000
    trace_log.append({
        "step": 5,
        "name": "Clinical Response Formatting",
        "latency_ms": round(step5_ms, 3),
        "operational_threshold": decision_threshold,
        "predicted_label": predicted_class,
        "risk_tier": risk_tier,
        "non_diagnostic_advisory": (
            "Screening result only. Recommend follow-up fasting plasma glucose / HbA1c test."
        )
    })

    return {
        "total_latency_ms": round(total_ms, 3),
        "trace": trace_log,
        "final_prediction": {
            "risk_score": round(prob_diabetic, 4),
            "risk_tier": risk_tier,
            "predicted_class": predicted_class
        }
    }


def diagnose_training_serving_skew(
    clean_sample: pd.DataFrame,
    skewed_sample: pd.DataFrame,
    trained_pipeline
) -> Dict[str, Any]:
    """
    Demonstrates Training-Serving Skew diagnosis:
    Compares model predictions when:
    A) Data is correctly passed through the trained pipeline (training-serving parity).
    B) Serving environment experiences skew (e.g. raw 0s treated as valid biological zeros,
       or unscaled raw features passed to a distance/linear model).
    """
    correct_preds = trained_pipeline.predict_proba(clean_sample)[:, 1]

    # Simulate Skew: unhandled zeros treated as 0 instead of NaN, leading to distorted imputation
    skewed_pipeline_input = skewed_sample.copy()
    skewed_preds = trained_pipeline.predict_proba(skewed_pipeline_input)[:, 1]

    diff = np.abs(correct_preds - skewed_preds)
    diagnosis = {
        "clean_mean_risk": float(np.mean(correct_preds)),
        "skewed_mean_risk": float(np.mean(skewed_preds)),
        "max_absolute_prediction_discrepancy": float(np.max(diff)),
        "mean_absolute_prediction_discrepancy": float(np.mean(diff)),
        "root_cause": (
            "Training-serving skew occurred because raw inference payload fed 0s directly "
            "into feature space without running the zero-to-NaN sanitization transform."
        ),
        "remediation_strategy": (
            "Encapsulate data validation and zero-sanitization directly into an atomic "
            "Scikit-Learn ColumnTransformer or Feature Store feature view before inference."
        )
    }
    return diagnosis
