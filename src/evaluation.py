"""
CO5: Model Evaluation, Cross-Validation, Calibration, Threshold Tuning, and Significance Testing.
"""
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, roc_curve,
    confusion_matrix, brier_score_loss
)
from scipy.stats import binomtest, chi2

from src.config import ARTIFACTS_DIR, RANDOM_STATE, CLINICAL_RECALL_TARGET


def run_nested_cv_benchmark(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict[str, Any],
    outer_k: int = 5,
    inner_k: int = 3
) -> Dict[str, Any]:
    """
    Performs Nested Cross-Validation (outer 5-fold, inner 3-fold)
    to calculate unbiased generalization error and quantify optimism bias.
    """
    outer_cv = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=RANDOM_STATE)

    outer_scores = []

    for train_idx, val_idx in outer_cv.split(X, y):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        grid = GridSearchCV(
            estimator,
            param_grid,
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=-1
        )
        grid.fit(X_tr, y_tr)
        best_model = grid.best_estimator_
        val_score = roc_auc_score(y_va, best_model.predict_proba(X_va)[:, 1])
        outer_scores.append(val_score)

    nested_mean = float(np.mean(outer_scores))
    nested_std = float(np.std(outer_scores))

    # Non-nested standard grid search on whole dataset
    standard_grid = GridSearchCV(
        estimator,
        param_grid,
        cv=outer_cv,
        scoring="roc_auc",
        n_jobs=-1
    )
    standard_grid.fit(X, y)
    non_nested_best = float(standard_grid.best_score_)
    optimism_bias = float(non_nested_best - nested_mean)

    return {
        "stratified_kfold_mean": round(nested_mean, 4),
        "std": round(nested_std, 4),
        "nested_cv_mean": round(nested_mean, 4),
        "non_nested_best": round(non_nested_best, 4),
        "optimism_bias": round(optimism_bias, 4)
    }


def compute_clinical_metrics_suite(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Computes comprehensive classification metrics with clinical emphasis on Recall & PR-AUC.
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)

    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recalls, precisions)

    # Specificity: TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "threshold": round(threshold, 3),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall_sensitivity": round(rec, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "clinical_note": (
            f"At threshold={threshold:.2f}, false negatives = {fn}. "
            "False negatives represent missed diabetic risk cases."
        )
    }


def sweep_classification_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    recall_target: float = CLINICAL_RECALL_TARGET,
    thresholds: np.ndarray = np.linspace(0.05, 0.95, 37)
) -> Tuple[pd.DataFrame, float]:
    """
    Sweeps decision threshold from 0.05 to 0.95 to locate optimal operating point
    achieving clinical recall >= recall_target.
    """
    records = []
    for t in thresholds:
        metrics = compute_clinical_metrics_suite(y_true, y_prob, threshold=t)
        cm = metrics["confusion_matrix"]
        records.append({
            "threshold": t,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall_sensitivity"],
            "f1_score": metrics["f1_score"],
            "TP": cm["TP"],
            "FP": cm["FP"],
            "TN": cm["TN"],
            "FN": cm["FN"]
        })

    df_thresh = pd.DataFrame(records)

    # Filter thresholds where recall >= recall_target
    eligible = df_thresh[df_thresh["recall"] >= recall_target]
    if not eligible.empty:
        # Pick the one with the highest precision among those meeting target recall
        best_row = eligible.loc[eligible["precision"].idxmax()]
        recommended_threshold = float(best_row["threshold"])
    else:
        best_row = df_thresh.loc[df_thresh["recall"].idxmax()]
        recommended_threshold = float(best_row["threshold"])

    # Plot Precision-Recall Trade-off across Thresholds
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(df_thresh["threshold"], df_thresh["recall"], label="Recall (Sensitivity)", color="#d95f02", lw=2)
    ax1.plot(df_thresh["threshold"], df_thresh["precision"], label="Precision (PPV)", color="#2b5c8f", lw=2)
    ax1.plot(df_thresh["threshold"], df_thresh["f1_score"], label="F1-Score", color="#7570b3", lw=1.8, linestyle="--")
    ax1.axvline(x=recommended_threshold, color="green", linestyle=":", lw=2,
                label=f"Recommended Threshold ({recommended_threshold:.2f})")
    ax1.axhline(y=recall_target, color="red", linestyle="--", alpha=0.7, label=f"Clinical Target Recall ({recall_target:.2f})")
    ax1.set_title("Clinical Threshold Tuning: Recall vs Precision Trade-off", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Decision Threshold")
    ax1.set_ylabel("Metric Value")
    ax1.legend()
    ax1.grid(True)

    # Ax2: False Negatives vs False Positives
    ax2.plot(df_thresh["threshold"], df_thresh["FN"], label="False Negatives (Missed Risk)", color="#e41a1c", lw=2)
    ax2.plot(df_thresh["threshold"], df_thresh["FP"], label="False Positives (Extra Screening)", color="#377eb8", lw=2)
    ax2.axvline(x=recommended_threshold, color="green", linestyle=":", lw=2, label=f"Selected Threshold ({recommended_threshold:.2f})")
    ax2.set_title("Diagnostic Error Breakdown: False Negatives vs False Positives", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Decision Threshold")
    ax2.set_ylabel("Case Count")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "threshold_tuning_curves.png", dpi=200)
    plt.close()

    return df_thresh, recommended_threshold


def calibrate_and_plot_reliability(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> Tuple[Any, Dict[str, Any]]:
    """
    Applies Platt Scaling (Sigmoid) and Isotonic Calibration.
    Compares Brier scores and produces Reliability Diagrams.
    """
    # Uncalibrated baseline
    prob_uncal = model.predict_proba(X_val)[:, 1]
    brier_uncal = brier_score_loss(y_val, prob_uncal)

    # 1. Platt Scaling (Sigmoid)
    cal_sigmoid = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    cal_sigmoid.fit(X_val, y_val)
    prob_sig = cal_sigmoid.predict_proba(X_val)[:, 1]
    brier_sig = brier_score_loss(y_val, prob_sig)

    # 2. Isotonic Calibration
    cal_iso = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    cal_iso.fit(X_val, y_val)
    prob_iso = cal_iso.predict_proba(X_val)[:, 1]
    brier_iso = brier_score_loss(y_val, prob_iso)

    # Reliability Curves
    frac_uncal, mean_uncal = calibration_curve(y_val, prob_uncal, n_bins=10)
    frac_sig, mean_sig = calibration_curve(y_val, prob_sig, n_bins=10)
    frac_iso, mean_iso = calibration_curve(y_val, prob_iso, n_bins=10)

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    plt.plot(mean_uncal, frac_uncal, "s-", color="#e41a1c", label=f"Uncalibrated (Brier={brier_uncal:.4f})")
    plt.plot(mean_sig, frac_sig, "o-", color="#2b5c8f", label=f"Platt Sigmoid (Brier={brier_sig:.4f})")
    plt.plot(mean_iso, frac_iso, "^-", color="#4daf4a", label=f"Isotonic (Brier={brier_iso:.4f})")
    plt.title("Reliability Diagram: Model Probability Calibration", fontsize=13, fontweight="bold")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives (Empirical Probability)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "calibration_reliability.png", dpi=200)
    plt.close()

    summary = {
        "brier_uncalibrated": round(brier_uncal, 4),
        "brier_sigmoid_platt": round(brier_sig, 4),
        "brier_isotonic": round(brier_iso, 4),
        "brier_improvement_percent": round((brier_uncal - brier_sig) / brier_uncal * 100, 2)
    }
    return cal_sigmoid, summary


def run_mcnemar_significance_test(
    y_true: np.ndarray,
    pred_model_a: np.ndarray,
    pred_model_b: np.ndarray,
    name_a: str = "Model A",
    name_b: str = "Model B"
) -> Dict[str, Any]:
    """
    Applies McNemar's test to determine whether predictions of two competing models
    differ with statistical significance using exact binomial or continuity-corrected chi2.
    """
    correct_a = (pred_model_a == y_true)
    correct_b = (pred_model_b == y_true)

    n00 = int(np.sum((correct_a) & (correct_b)))
    n01 = int(np.sum((correct_a) & (~correct_b)))  # A correct, B wrong (b)
    n10 = int(np.sum((~correct_a) & (correct_b)))  # A wrong, B correct (c)
    n11 = int(np.sum((~correct_a) & (~correct_b)))

    b = n01
    c = n10
    total_discordant = b + c

    if total_discordant == 0:
        p_val = 1.0
        stat = 0.0
    elif total_discordant < 25:
        # Exact binomial test
        res = binomtest(min(b, c), total_discordant, 0.5, alternative="two-sided")
        p_val = float(res.pvalue)
        stat = float(min(b, c))
    else:
        # Edwards continuity-corrected chi-squared test
        stat = float(((abs(b - c) - 1.0) ** 2) / total_discordant)
        p_val = float(chi2.sf(stat, df=1))

    table = [[n00, n01], [n10, n11]]
    return {
        "model_a": name_a,
        "model_b": name_b,
        "contingency_table": table,
        "n_discordant_pairs": total_discordant,
        "statistic": round(stat, 4),
        "p_value": round(p_val, 4),
        "statistically_significant": bool(p_val < 0.05),
        "interpretation": (
            f"Difference between {name_a} and {name_b} is "
            f"{'statistically significant (p < 0.05)' if p_val < 0.05 else 'not statistically significant (p >= 0.05)'}."
        )
    }
