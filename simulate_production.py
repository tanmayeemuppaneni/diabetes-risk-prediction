"""
CO6: Production Batch Scoring and Data Drift Simulation Script.
"""
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, FEATURE_COLUMNS, RANDOM_STATE
from src.data_loader import load_raw_data, clean_dataset
from src.pipeline import build_production_pipeline
from src.monitoring import generate_psi_report, plot_drift_dashboard
from src.export_onnx import load_pipeline_joblib


def simulate_production_monitoring():
    print("=" * 70)
    print("SIMULATING PRODUCTION BATCH SCORING & PSI DRIFT OBSERVABILITY")
    print("=" * 70)

    # 1. Load baseline training reference data
    df_raw = load_raw_data()
    reference_data = clean_dataset(df_raw)[FEATURE_COLUMNS]

    # 2. Simulate Batch 1: In-distribution normal production traffic
    np.random.seed(RANDOM_STATE)
    batch_normal = reference_data.sample(n=120, replace=True, random_state=101).copy()
    # Add minor Gaussian noise
    for col in FEATURE_COLUMNS:
        batch_normal[col] += np.random.normal(0, batch_normal[col].std() * 0.05, size=len(batch_normal))

    print("\n[BATCH 1: Standard In-Distribution Clinic Traffic]")
    psi_normal = generate_psi_report(reference_data, batch_normal, FEATURE_COLUMNS)
    print(psi_normal.to_string(index=False))

    # 3. Simulate Batch 2: Shifted high-risk / older referral population (Data Drift)
    batch_drifted = reference_data.sample(n=120, replace=True, random_state=202).copy()
    # Induce demographic & physiological distribution shifts
    batch_drifted["Age"] += np.random.normal(16.0, 3.0, size=len(batch_drifted))
    batch_drifted["Glucose"] += np.random.normal(32.0, 8.0, size=len(batch_drifted))
    batch_drifted["BMI"] += np.random.normal(6.5, 2.0, size=len(batch_drifted))

    print("\n[BATCH 2: High-Risk Outpatient Cohort (Simulated Shift)]")
    psi_drifted = generate_psi_report(reference_data, batch_drifted, FEATURE_COLUMNS)
    print(psi_drifted.to_string(index=False))

    # 4. Generate Production Observability Dashboard
    plot_drift_dashboard(reference_data, batch_drifted, psi_drifted)
    print(f"\nDrift Observability Dashboard saved to: {ARTIFACTS_DIR / 'drift_dashboard.png'}")

    # 5. Batch Scoring with Trained Pipeline
    pipeline = load_pipeline_joblib()
    batch_preds = pipeline.predict_proba(batch_drifted)[:, 1]
    batch_drifted["predicted_risk_probability"] = np.round(batch_preds, 4)
    batch_drifted["predicted_risk_class"] = (batch_preds >= 0.35).astype(int)
    batch_drifted["risk_tier"] = [
        "Low" if p < 0.30 else ("Moderate" if p < 0.60 else "High")
        for p in batch_preds
    ]

    out_csv = ARTIFACTS_DIR / "batch_production_scored.csv"
    batch_drifted.to_csv(out_csv, index=False)
    print(f"Scored batch saved to: {out_csv}")

    # Retraining trigger check
    drifted_count = (psi_drifted["status"] == "DRIFTED").sum()
    if drifted_count > 0:
        drifted_features = psi_drifted[psi_drifted["status"] == "DRIFTED"]["feature"].tolist()
        print(f"\n[ALERT] Retraining trigger activated! Detected significant drift in: {drifted_features}")
    else:
        print("\nAll feature distributions within stability boundaries.")


if __name__ == "__main__":
    simulate_production_monitoring()
