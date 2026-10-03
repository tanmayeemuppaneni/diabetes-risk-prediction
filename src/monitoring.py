"""
CO6: Production Monitoring and Population Stability Index (PSI) Drift Detection Module.
"""
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import ARTIFACTS_DIR, PSI_STABLE_THRESHOLD, PSI_DRIFT_THRESHOLD


def calculate_feature_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4
) -> float:
    """
    Computes the Population Stability Index (PSI) between baseline (expected)
    and production (actual) distributions for a continuous feature.

    Formula:
      PSI = sum( (Actual% - Expected%) * ln(Actual% / Expected%) )
    """
    # Filter out NaNs if any
    exp_clean = expected[~np.isnan(expected)]
    act_clean = actual[~np.isnan(actual)]

    if len(exp_clean) == 0 or len(act_clean) == 0:
        return 0.0

    # Determine quantile bins from reference/expected distribution
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(exp_clean, quantiles)
    bin_edges = np.unique(bin_edges)  # remove duplicate edges

    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    exp_counts, _ = np.histogram(exp_clean, bins=bin_edges)
    act_counts, _ = np.histogram(act_clean, bins=bin_edges)

    exp_pct = exp_counts / len(exp_clean)
    act_pct = act_counts / len(act_clean)

    # Apply epsilon smoothing to prevent division by zero or ln(0)
    exp_pct = np.clip(exp_pct, epsilon, 1.0)
    act_pct = np.clip(act_pct, epsilon, 1.0)

    # Re-normalize after clipping
    exp_pct = exp_pct / np.sum(exp_pct)
    act_pct = act_pct / np.sum(act_pct)

    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi_val)


def generate_psi_report(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = PSI_DRIFT_THRESHOLD
) -> pd.DataFrame:
    """
    Generates a PSI report across all monitored features.
    Categorizes status into:
      - STABLE (PSI < 0.10)
      - MODERATE_SHIFT (0.10 <= PSI < 0.20)
      - DRIFTED (PSI >= 0.20)
    """
    records = []
    for col in feature_cols:
        if col in reference_df.columns and col in production_df.columns:
            psi = calculate_feature_psi(reference_df[col].values, production_df[col].values)
            if psi < PSI_STABLE_THRESHOLD:
                status = "STABLE"
                action = "None"
            elif psi < threshold:
                status = "MODERATE_SHIFT"
                action = "Monitor Closely"
            else:
                status = "DRIFTED"
                action = "Trigger Model Retraining"

            records.append({
                "feature": col,
                "PSI": round(psi, 4),
                "status": status,
                "recommended_action": action
            })

    report_df = pd.DataFrame(records).sort_values(by="PSI", ascending=False)
    n_drifted = (report_df["status"] == "DRIFTED").sum()
    n_stable = (report_df["status"] == "STABLE").sum()
    print(f"PSI Report Summary: {n_drifted} DRIFTED, {n_stable} STABLE (Threshold = {threshold})")
    return report_df


def plot_drift_dashboard(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    psi_report_df: pd.DataFrame
) -> None:
    """
    Plots side-by-side distribution comparisons for top drifted vs stable features.
    """
    top_features = psi_report_df["feature"].head(4).tolist()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for i, col in enumerate(top_features):
        psi = psi_report_df.loc[psi_report_df["feature"] == col, "PSI"].values[0]
        status = psi_report_df.loc[psi_report_df["feature"] == col, "status"].values[0]

        sns.kdeplot(reference_df[col], ax=axes[i], label="Baseline (Reference)", color="#2b5c8f", lw=2)
        sns.kdeplot(production_df[col], ax=axes[i], label="Production (Incoming)", color="#d95f02", lw=2, linestyle="--")
        axes[i].set_title(f"{col} | PSI = {psi:.4f} ({status})", fontsize=11, fontweight="bold")
        axes[i].legend()
        axes[i].grid(True)

    plt.suptitle("Production Data Drift Observability Dashboard (PSI Analysis)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "drift_dashboard.png", dpi=200)
    plt.close()
