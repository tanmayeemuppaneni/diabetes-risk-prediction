"""
Configuration settings, constants, and feature definitions for Diabetes Risk Prediction.
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "diabetes.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Set Matplotlib config directory to avoid permission issues
MPL_DIR = ARTIFACTS_DIR / ".matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_DIR)

MODEL_JOBLIB_PATH = ARTIFACTS_DIR / "diabetes_model.joblib"
MODEL_ONNX_PATH = ARTIFACTS_DIR / "diabetes_model.onnx"
CALIBRATED_MODEL_PATH = ARTIFACTS_DIR / "diabetes_calibrated_model.joblib"

# Random Seed for Reproducibility
RANDOM_STATE = 42

# Feature Definitions
FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

TARGET_COLUMN = "Outcome"

# Biological features where a value of 0 is physiologically invalid (represents missing data)
PHYSIOLOGICAL_ZERO_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Clinical Screening Disclaimer
CLINICAL_DISCLAIMER = (
    "CAUTION: This system is an algorithmic risk-screening and triage-support aid. "
    "It is NOT a certified medical diagnostic device. Predictions must be validated by a licensed physician."
)

# High recall target to minimize dangerous False Negatives (Type II errors)
CLINICAL_RECALL_TARGET = 0.90

# PSI Drift Monitoring Thresholds
PSI_STABLE_THRESHOLD = 0.10
PSI_DRIFT_THRESHOLD = 0.20
