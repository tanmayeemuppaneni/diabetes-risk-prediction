"""
CO2: Linear Supervised-Learning Models, Regularisation, and Gradient Descent.
"""
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

from src.config import ARTIFACTS_DIR, RANDOM_STATE


class ScratchLinearRegressionGD:
    """
    Scratch implementation of Linear Regression via Batch & Mini-Batch Gradient Descent.
    """
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000, batch_size: int = None):
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.loss_history: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        np.random.seed(RANDOM_STATE)

        for epoch in range(self.epochs):
            if self.batch_size is None or self.batch_size >= n_samples:
                # Batch Gradient Descent
                y_pred = np.dot(X, self.weights) + self.bias
                error = y_pred - y
                dw = (2 / n_samples) * np.dot(X.T, error)
                db = (2 / n_samples) * np.sum(error)
                self.weights -= self.lr * dw
                self.bias -= self.lr * db
            else:
                # Mini-Batch Gradient Descent
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]

                for start_idx in range(0, n_samples, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, n_samples)
                    X_b = X_shuffled[start_idx:end_idx]
                    y_b = y_shuffled[start_idx:end_idx]
                    b_size = len(y_b)

                    y_pred_b = np.dot(X_b, self.weights) + self.bias
                    error_b = y_pred_b - y_b
                    dw = (2 / b_size) * np.dot(X_b.T, error_b)
                    db = (2 / b_size) * np.sum(error_b)
                    self.weights -= self.lr * dw
                    self.bias -= self.lr * db

            current_loss = mean_squared_error(y, self.predict(X))
            self.loss_history.append(current_loss)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias


def run_gradient_descent_benchmark(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> Dict[str, Any]:
    """
    Fits Batch GD, Mini-Batch GD, and Scikit-Learn Linear Regression.
    Generates MSE loss comparison curves.
    """
    # Standardize input features for stable gradient descent
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_va_scaled = scaler.transform(X_val)

    # 1. Batch GD
    bgd = ScratchLinearRegressionGD(learning_rate=0.02, epochs=800, batch_size=None)
    bgd.fit(X_tr_scaled, y_train)
    bgd_val_rmse = np.sqrt(mean_squared_error(y_val, bgd.predict(X_va_scaled)))

    # 2. Mini-Batch GD (batch_size=32)
    mbgd = ScratchLinearRegressionGD(learning_rate=0.02, epochs=800, batch_size=32)
    mbgd.fit(X_tr_scaled, y_train)
    mbgd_val_rmse = np.sqrt(mean_squared_error(y_val, mbgd.predict(X_va_scaled)))

    # 3. Sklearn Closed-Form Linear Regression
    sk_lr = LinearRegression()
    sk_lr.fit(X_tr_scaled, y_train)
    sk_val_rmse = np.sqrt(mean_squared_error(y_val, sk_lr.predict(X_va_scaled)))

    # Plot MSE Loss History
    plt.figure(figsize=(9, 5))
    plt.plot(bgd.loss_history, label=f"Batch GD (Final Val RMSE={bgd_val_rmse:.3f})", color="#1f77b4", lw=2)
    plt.plot(mbgd.loss_history, label=f"Mini-Batch GD (b=32) (Final Val RMSE={mbgd_val_rmse:.3f})", color="#ff7f0e", lw=1.8, alpha=0.85)
    plt.axhline(y=mean_squared_error(y_train, sk_lr.predict(X_tr_scaled)), color="green", linestyle="--", label=f"Sklearn OLS (Val RMSE={sk_val_rmse:.3f})")
    plt.title("Gradient Descent Optimization: Batch vs Mini-Batch MSE Loss Convergence", fontsize=12, fontweight="bold")
    plt.xlabel("Epochs")
    plt.ylabel("Training Mean Squared Error (MSE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "gd_loss_curves.png", dpi=200)
    plt.close()

    return {
        "batch_gd_val_rmse": float(bgd_val_rmse),
        "minibatch_gd_val_rmse": float(mbgd_val_rmse),
        "sklearn_ols_val_rmse": float(sk_val_rmse),
        "rmse_parity_within_0_05": bool(abs(mbgd_val_rmse - sk_val_rmse) < 0.05)
    }


def compare_regularisation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    C_values: List[float] = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compares Ridge (L2), Lasso (L1), and Elastic Net across C values.
    Returns comparison DataFrame with validation accuracy, recall, and sparsity (non-zero weights).
    """
    results = []
    lasso_coef_paths = []
    ridge_coef_paths = []

    for C in C_values:
        # 1. Ridge (L2 penalty)
        ridge = LogisticRegression(
            penalty="l2", C=C, solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE
        )
        ridge.fit(X_train, y_train)
        ridge_pred = ridge.predict(X_val)
        ridge_nonzero = int(np.sum(np.abs(ridge.coef_) > 1e-4))
        ridge_coef_paths.append(ridge.coef_[0].copy())

        # 2. Lasso (L1 penalty)
        lasso = LogisticRegression(
            penalty="l1", C=C, solver="saga", max_iter=2500, random_state=RANDOM_STATE
        )
        lasso.fit(X_train, y_train)
        lasso_pred = lasso.predict(X_val)
        lasso_nonzero = int(np.sum(np.abs(lasso.coef_) > 1e-4))
        lasso_coef_paths.append(lasso.coef_[0].copy())

        # 3. Elastic Net (L1 ratio = 0.5)
        elastic = LogisticRegression(
            penalty="elasticnet", l1_ratio=0.5, C=C, solver="saga", max_iter=2500, random_state=RANDOM_STATE
        )
        elastic.fit(X_train, y_train)
        elastic_pred = elastic.predict(X_val)
        elastic_nonzero = int(np.sum(np.abs(elastic.coef_) > 1e-4))

        results.append({
            "C": C,
            "Ridge_val_acc": accuracy_score(y_val, ridge_pred),
            "Ridge_val_rec": recall_score(y_val, ridge_pred),
            "Ridge_nonzero": ridge_nonzero,
            "Lasso_val_acc": accuracy_score(y_val, lasso_pred),
            "Lasso_val_rec": recall_score(y_val, lasso_pred),
            "Lasso_nonzero": lasso_nonzero,
            "ElasticNet_val_acc": accuracy_score(y_val, elastic_pred),
            "ElasticNet_val_rec": recall_score(y_val, elastic_pred),
            "ElasticNet_nonzero": elastic_nonzero
        })

    res_df = pd.DataFrame(results)

    # Plot Shrinkage Paths and Sparsity Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Ax1: Non-zero features (Sparsity)
    ax1.plot(res_df["C"], res_df["Lasso_nonzero"], marker="o", label="Lasso (L1) Non-Zero Count", color="#d95f02", lw=2)
    ax1.plot(res_df["C"], res_df["Ridge_nonzero"], marker="s", label="Ridge (L2) Non-Zero Count", color="#2b5c8f", lw=2)
    ax1.plot(res_df["C"], res_df["ElasticNet_nonzero"], marker="^", label="ElasticNet Non-Zero Count", color="#7570b3", lw=2)
    ax1.set_xscale("log")
    ax1.set_xlabel("Regularization Parameter C (1 / Lambda)")
    ax1.set_ylabel("Count of Non-Zero Feature Coefficients")
    ax1.set_title("Regularization & Sparsity: L1 Induces Exact Zero Weights", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True)

    # Ax2: Validation Recall Comparison
    ax2.plot(res_df["C"], res_df["Lasso_val_rec"], marker="o", label="Lasso (L1) Recall", color="#d95f02", lw=2)
    ax2.plot(res_df["C"], res_df["Ridge_val_rec"], marker="s", label="Ridge (L2) Recall", color="#2b5c8f", lw=2)
    ax2.plot(res_df["C"], res_df["ElasticNet_val_rec"], marker="^", label="ElasticNet Recall", color="#7570b3", lw=2)
    ax2.set_xscale("log")
    ax2.set_xlabel("Regularization Parameter C (1 / Lambda)")
    ax2.set_ylabel("Validation Recall (Sensitivity)")
    ax2.set_title("Validation Recall Across Regularization Strengths", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "regularization_paths.png", dpi=200)
    plt.close()

    # Feature Scaling Comparison (demonstrating solver efficiency with scaling)
    unscaled_lr = LogisticRegression(solver="liblinear", max_iter=200, random_state=RANDOM_STATE)
    unscaled_lr.fit(X_train, y_train)

    scaled_lr = LogisticRegression(solver="liblinear", max_iter=200, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    scaled_lr.fit(scaler.fit_transform(X_train), y_train)

    scaling_comparison = {
        "unscaled_iterations": int(unscaled_lr.n_iter_[0]),
        "scaled_iterations": int(scaled_lr.n_iter_[0]),
        "iterations_reduction_factor": float(unscaled_lr.n_iter_[0] / max(1, scaled_lr.n_iter_[0]))
    }

    return res_df, scaling_comparison
