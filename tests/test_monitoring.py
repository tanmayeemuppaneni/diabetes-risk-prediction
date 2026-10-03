"""
Unit tests for PSI calculation and data drift detection.
"""
import pytest
import numpy as np
import pandas as pd

from src.monitoring import calculate_feature_psi, generate_psi_report


def test_psi_identical_distributions():
    np.random.seed(42)
    baseline = np.random.normal(100, 15, size=1000)
    production = np.random.normal(100, 15, size=1000)

    psi = calculate_feature_psi(baseline, production, num_bins=10)
    assert psi < 0.10, f"Expected stable PSI < 0.10, got {psi}"


def test_psi_drifted_distribution():
    np.random.seed(42)
    baseline = np.random.normal(100, 15, size=1000)
    # Substantial distribution shift
    production = np.random.normal(140, 20, size=1000)

    psi = calculate_feature_psi(baseline, production, num_bins=10)
    assert psi >= 0.20, f"Expected drifted PSI >= 0.20, got {psi}"


def test_generate_psi_report_categorization():
    np.random.seed(42)
    ref_df = pd.DataFrame({
        "FeatureA": np.random.normal(50, 10, size=500),
        "FeatureB": np.random.normal(20, 5, size=500)
    })
    # FeatureA remains stable, FeatureB drifts heavily
    prod_df = pd.DataFrame({
        "FeatureA": np.random.normal(50, 10, size=500),
        "FeatureB": np.random.normal(45, 10, size=500)
    })

    report = generate_psi_report(ref_df, prod_df, ["FeatureA", "FeatureB"], threshold=0.20)
    status_map = dict(zip(report["feature"], report["status"]))

    assert status_map["FeatureA"] == "STABLE"
    assert status_map["FeatureB"] == "DRIFTED"
