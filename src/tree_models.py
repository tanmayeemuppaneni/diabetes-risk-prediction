"""
CO3: Tree-Based Supervised-Learning Models, Ensembles, SVM, and Explainability.
"""
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance
import shap

from src.config import ARTIFACTS_DIR, RANDOM_STATE, FEATURE_COLUMNS

# Graceful optional imports for XGBoost and LightGBM (which may require external libomp runtime on macOS)
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False


def sweep_decision_tree_depth(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    max_depths: List[int] = list(range(1, 16))
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Sweeps Decision Tree max_depth from 1 to 15.
    Compares Gini vs Entropy and analyzes underfitting vs overfitting.
    """
    records = []
    for d in max_depths:
        # Gini Tree
        dt_gini = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=RANDOM_STATE)
        dt_gini.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, dt_gini.predict(X_train))
        val_acc = accuracy_score(y_val, dt_gini.predict(X_val))
        val_rec = recall_score(y_val, dt_gini.predict(X_val))

        # Entropy Tree
        dt_ent = DecisionTreeClassifier(criterion="entropy", max_depth=d, random_state=RANDOM_STATE)
        dt_ent.fit(X_train, y_train)
        ent_val_acc = accuracy_score(y_val, dt_ent.predict(X_val))

        records.append({
            "depth": d,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_recall": val_rec,
            "entropy_val_acc": ent_val_acc,
            "n_leaves": dt_gini.get_n_leaves()
        })

    df_sweep = pd.DataFrame(records)

    # Cost-Complexity Pruning (CCP) Path
    dt_full = DecisionTreeClassifier(random_state=RANDOM_STATE)
    path = dt_full.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas[::max(1, len(path.ccp_alphas) // 15)]

    pruned_scores = []
    for alpha in ccp_alphas:
        dt_pruned = DecisionTreeClassifier(random_state=RANDOM_STATE, ccp_alpha=alpha)
        dt_pruned.fit(X_train, y_train)
        pruned_scores.append({
            "alpha": alpha,
            "val_acc": accuracy_score(y_val, dt_pruned.predict(X_val)),
            "val_recall": recall_score(y_val, dt_pruned.predict(X_val))
        })
    df_pruned = pd.DataFrame(pruned_scores)
    best_pruned = df_pruned.loc[df_pruned["val_acc"].idxmax()]

    # Plot Depth Sweep & Pruning Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Ax1: Depth Sweep
    ax1.plot(df_sweep["depth"], df_sweep["train_acc"], "o-", label="Train Accuracy (Gini)", color="#2b5c8f")
    ax1.plot(df_sweep["depth"], df_sweep["val_acc"], "s-", label="Validation Accuracy (Gini)", color="#d95f02")
    ax1.plot(df_sweep["depth"], df_sweep["entropy_val_acc"], "^--", label="Validation Accuracy (Entropy)", color="#7570b3")
    best_depth_row = df_sweep.loc[df_sweep["val_acc"].idxmax()]
    ax1.axvline(x=best_depth_row["depth"], color="red", linestyle=":", label=f"Optimal Depth = {int(best_depth_row['depth'])}")
    ax1.set_title("Decision Tree: Depth Sweep (Bias-Variance Tradeoff)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Max Depth")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    # Ax2: Cost Complexity Pruning Path
    ax2.plot(df_pruned["alpha"], df_pruned["val_acc"], "o-", label="Validation Accuracy vs Alpha", color="#1b9e77")
    ax2.set_xscale("log")
    ax2.set_title("Cost-Complexity Pruning (CCP) Path", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Effective Alpha")
    ax2.set_ylabel("Validation Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "dt_depth_sweep.png", dpi=200)
    plt.close()

    summary = {
        "optimal_depth": int(best_depth_row["depth"]),
        "optimal_depth_val_acc": float(best_depth_row["val_acc"]),
        "full_tree_overfit_train_acc": float(df_sweep.iloc[-1]["train_acc"]),
        "full_tree_val_acc": float(df_sweep.iloc[-1]["val_acc"]),
        "optimal_ccp_alpha": float(best_pruned["alpha"]),
        "pruned_val_acc": float(best_pruned["val_acc"])
    }
    return df_sweep, summary


def benchmark_random_forest_oob(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_estimators_list: List[int] = [10, 25, 50, 100, 150, 200, 300]
) -> Tuple[pd.DataFrame, RandomForestClassifier]:
    """
    Evaluates Random Forest Out-Of-Bag (OOB) error convergence vs held-out validation accuracy
    across different numbers of trees.
    """
    records = []
    dt_baseline = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)
    dt_baseline.fit(X_train, y_train)
    dt_val_acc = accuracy_score(y_val, dt_baseline.predict(X_val))

    best_rf = None
    best_val_score = -1.0

    for n_trees in n_estimators_list:
        rf = RandomForestClassifier(
            n_estimators=n_trees,
            oob_score=True,
            bootstrap=True,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        val_pred = rf.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        val_rec = recall_score(y_val, val_pred)
        oob_acc = rf.oob_score_

        if val_acc > best_val_score:
            best_val_score = val_acc
            best_rf = rf

        records.append({
            "n_estimators": n_trees,
            "RF_val_acc": val_acc,
            "RF_val_recall": val_rec,
            "OOB_acc": oob_acc,
            "DT_val_acc": dt_val_acc
        })

    df_rf = pd.DataFrame(records)

    # Plot OOB Error Convergence
    plt.figure(figsize=(9, 5))
    plt.plot(df_rf["n_estimators"], df_rf["OOB_acc"], "o-", label="OOB Accuracy", color="#2b5c8f", lw=2)
    plt.plot(df_rf["n_estimators"], df_rf["RF_val_acc"], "s-", label="Validation Accuracy (RF)", color="#d95f02", lw=2)
    plt.axhline(y=dt_val_acc, color="gray", linestyle="--", label=f"Single Decision Tree Baseline ({dt_val_acc:.3f})")
    plt.title("Random Forest: OOB Error Convergence vs Validation Performance", fontsize=12, fontweight="bold")
    plt.xlabel("Number of Estimators (Trees)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "rf_oob_convergence.png", dpi=200)
    plt.close()

    return df_rf, best_rf


def benchmark_all_supervised_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Trains and compares:
    1. Logistic Regression
    2. Support Vector Machine (Linear Kernel)
    3. Support Vector Machine (RBF Kernel)
    4. Decision Tree
    5. Random Forest
    6. AdaBoost
    7. Gradient Boosting (GBDT)
    8. Histogram Gradient Boosting (Modern Histogram-based boosting)
    (And XGBoost / LightGBM if runtime OpenMP libraries are present)
    """
    models = {
        "Logistic Regression": LogisticRegression(C=0.5, max_iter=1000, random_state=RANDOM_STATE),
        "SVM (Linear)": SVC(kernel="linear", C=0.5, probability=True, random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_features="sqrt", random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=100, max_depth=3, random_state=RANDOM_STATE)
    }

    if HAS_XGB:
        models["XGBoost"] = xgb.XGBClassifier(n_estimators=100, max_depth=3, eval_metric="logloss", random_state=RANDOM_STATE)
    if HAS_LGB:
        models["LightGBM"] = lgb.LGBMClassifier(n_estimators=100, max_depth=3, verbose=-1, random_state=RANDOM_STATE)

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_prob = model.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, val_pred)
        prec = precision_score(y_val, val_pred, zero_division=0)
        rec = recall_score(y_val, val_pred, zero_division=0)
        f1 = f1_score(y_val, val_pred, zero_division=0)
        roc = roc_auc_score(y_val, val_prob)

        trained_models[name] = model
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc
        })

    df_benchmark = pd.DataFrame(results).sort_values(by="Recall", ascending=False)
    return df_benchmark, trained_models


def compute_explainability_and_shap(
    model,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: List[str]
) -> Dict[str, Any]:
    """
    Computes Gini importance, Permutation Importance, and SHAP values for tree models.
    Generates SHAP summary and feature importance comparison plots.
    """
    # 1. Gini Importance (MDI)
    mdi_importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

    # 2. Permutation Importance
    perm = permutation_importance(model, X_val, y_val, scoring="recall", n_repeats=10, random_state=RANDOM_STATE)
    perm_importance = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)

    # 3. SHAP Values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    if isinstance(shap_values, list):
        shap_matrix = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_matrix = shap_values[:, :, 1]
    else:
        shap_matrix = shap_values

    # Plot SHAP summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_matrix, X_val, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance (Impact on Diabetes Risk)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Compare Gini vs Permutation Importance Bar Chart
    comp_df = pd.DataFrame({
        "Gini Importance (MDI)": mdi_importance,
        "Permutation Importance": perm_importance
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    comp_df.plot(kind="barh", ax=ax, color=["#2b5c8f", "#d95f02"], width=0.7)
    ax.set_title("Feature Importance: Gini Impurity (MDI) vs Permutation Importance", fontsize=12, fontweight="bold")
    ax.set_xlabel("Relative Importance Score")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "feature_importance_comparison.png", dpi=200)
    plt.close()

    mean_abs_shap = pd.Series(np.abs(shap_matrix).mean(axis=0), index=feature_names).sort_values(ascending=False)

    return {
        "top_features_shap": mean_abs_shap.head(5).to_dict(),
        "top_features_gini": mdi_importance.head(5).to_dict(),
        "top_features_permutation": perm_importance.head(5).to_dict()
    }
