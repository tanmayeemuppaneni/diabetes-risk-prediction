"""
CO1: Exploratory Data Analysis and Visualisation Module.
"""
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import ARTIFACTS_DIR, FEATURE_COLUMNS, TARGET_COLUMN, PHYSIOLOGICAL_ZERO_COLUMNS


def run_eda(df_raw: pd.DataFrame, df_cleaned: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform statistical analysis and generate publication-quality figures for EDA.
    """
    sns.set_theme(style="whitegrid", palette="muted")
    stats = {}

    # 1. Missingness & Zero Analysis
    zero_counts = {
        col: int((df_raw[col] == 0).sum())
        for col in PHYSIOLOGICAL_ZERO_COLUMNS
    }
    stats["zero_counts"] = zero_counts

    fig, ax = plt.subplots(figsize=(8, 4))
    zero_df = pd.DataFrame({
        "Feature": list(zero_counts.keys()),
        "Count": list(zero_counts.values()),
        "Percentage": [v / len(df_raw) * 100 for v in zero_counts.values()]
    })
    sns.barplot(data=zero_df, x="Feature", y="Percentage", hue="Feature", ax=ax, palette="Reds_r", legend=False)
    ax.set_title("Physiologically Impossible Zero Values (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Missing / Zero Percentage (%)")
    ax.set_xlabel("Clinical Measurement")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "missingness_analysis.png", dpi=200)
    plt.close()

    # 2. Histograms of Clinical Features
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for idx, col in enumerate(FEATURE_COLUMNS):
        sns.histplot(
            data=df_cleaned,
            x=col,
            hue=TARGET_COLUMN,
            kde=True,
            ax=axes[idx],
            bins=20,
            palette=["#2b5c8f", "#d95f02"]
        )
        axes[idx].set_title(f"Distribution: {col}", fontsize=11, fontweight="bold")
    plt.suptitle("Clinical Feature Distributions Stratified by Diabetes Outcome (0=Non-diabetic, 1=Diabetic)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "eda_distributions.png", dpi=200)
    plt.close()

    # 3. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = df_cleaned.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8}
    )
    plt.title("Pearson Correlation Matrix (with Zeroes Treated as Missing)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()

    # 4. Boxplots for Clinical Outliers and Separability
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    df_box = df_cleaned.copy()
    df_box["Outcome_Label"] = df_box[TARGET_COLUMN].map({0: "Non-Diabetic", 1: "Diabetic"})
    for idx, col in enumerate(FEATURE_COLUMNS):
        sns.boxplot(
            data=df_box,
            x="Outcome_Label",
            y=col,
            hue="Outcome_Label",
            ax=axes[idx],
            palette=["#6baed6", "#fd8d3c"],
            legend=False
        )
        axes[idx].set_title(f"{col} by Outcome", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("")
    plt.suptitle("Clinical Measurement Boxplots by Outcome", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "eda_boxplots.png", dpi=200)
    plt.close()

    # Save summary stats
    stats["class_balance"] = df_raw[TARGET_COLUMN].value_counts().to_dict()
    stats["top_correlations_with_target"] = (
        corr[TARGET_COLUMN].drop(TARGET_COLUMN).abs().sort_values(ascending=False).to_dict()
    )
    return stats


if __name__ == "__main__":
    from src.data_loader import load_raw_data, clean_dataset
    raw = load_raw_data()
    clean = clean_dataset(raw)
    stats = run_eda(raw, clean)
    print("EDA Complete. Artifacts saved. Top correlations with target:")
    for k, v in stats["top_correlations_with_target"].items():
        print(f"  {k}: {v:.3f}")
