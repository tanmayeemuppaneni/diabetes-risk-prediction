"""
Unit and integration tests for data ingestion, cleaning, and preprocessing pipeline.
"""
import pytest
import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, TARGET_COLUMN, PHYSIOLOGICAL_ZERO_COLUMNS
from src.data_loader import load_raw_data, clean_dataset, get_stratified_splits
from src.pipeline import build_production_pipeline, PhysiologicalZeroSanitizer


def test_data_ingestion_and_schema():
    df = load_raw_data()
    assert df.shape[0] == 768, f"Expected 768 rows, got {df.shape[0]}"
    assert df.shape[1] == 9, f"Expected 9 columns, got {df.shape[1]}"
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        assert col in df.columns, f"Column {col} missing from raw dataset"


def test_zero_to_nan_sanitization():
    raw_df = pd.DataFrame({
        "Pregnancies": [2, 0],
        "Glucose": [120, 0],
        "BloodPressure": [70, 0],
        "SkinThickness": [20, 0],
        "Insulin": [80, 0],
        "BMI": [25.0, 0],
        "DiabetesPedigreeFunction": [0.4, 0.5],
        "Age": [30, 25]
    })
    sanitizer = PhysiologicalZeroSanitizer()
    clean_df = sanitizer.transform(raw_df)

    # Pregnancies can legitimately be 0, but physiological clinical metrics cannot
    assert clean_df["Pregnancies"].iloc[1] == 0
    for col in PHYSIOLOGICAL_ZERO_COLUMNS:
        assert np.isnan(clean_df[col].iloc[1]), f"Feature {col} should have been converted to NaN"


def test_pipeline_transform_no_nans():
    df_raw = load_raw_data()
    df_clean = clean_dataset(df_raw)
    X_train, y_train, X_val, y_val, _, _ = get_stratified_splits(df_clean)

    pipe = build_production_pipeline()
    pipe.fit(X_train, y_train)

    prep = pipe.named_steps["preprocessor"]
    X_val_trans = prep.transform(X_val)

    assert not np.isnan(X_val_trans).any(), "Found NaNs in transformed validation output"
    assert X_val_trans.shape[0] == len(X_val)
    assert X_val_trans.shape[1] >= len(FEATURE_COLUMNS)


def test_pipeline_predictions_in_bounds():
    df_raw = load_raw_data()
    df_clean = clean_dataset(df_raw)
    X_train, y_train, X_val, y_val, _, _ = get_stratified_splits(df_clean)

    pipe = build_production_pipeline()
    pipe.fit(X_train, y_train)

    probs = pipe.predict_proba(X_val)[:, 1]
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0), "Predicted probabilities out of [0, 1] range"
    preds = pipe.predict(X_val)
    assert set(np.unique(preds)).issubset({0, 1}), "Predictions must be binary {0, 1}"
