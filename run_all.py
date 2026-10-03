"""
Orchestration Script: Runs the complete Diabetes Risk Prediction pipeline end-to-end (CO1 - CO6).
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import (
    RAW_DATA_PATH, ARTIFACTS_DIR, FEATURE_COLUMNS,
    TARGET_COLUMN, MODEL_JOBLIB_PATH, MODEL_ONNX_PATH,
    CALIBRATED_MODEL_PATH, CLINICAL_RECALL_TARGET
)
from src.data_loader import load_raw_data, clean_dataset, get_stratified_splits, summarise_dataset
from src.eda import run_eda
from src.lifecycle import LIFECYCLE_STAGES, trace_prediction_request, diagnose_training_serving_skew
from src.linear_models import run_gradient_descent_benchmark, compare_regularisation
from src.tree_models import (
    sweep_decision_tree_depth, benchmark_random_forest_oob,
    benchmark_all_supervised_models, compute_explainability_and_shap
)
from src.unsupervised import (
    run_kmeans_sweep, run_hierarchical_and_dendrogram, run_dbscan_analysis,
    run_dimensionality_reduction_dashboard, run_isolation_forest_anomaly_detection
)
from src.evaluation import (
    run_nested_cv_benchmark, compute_clinical_metrics_suite,
    sweep_classification_threshold, calibrate_and_plot_reliability,
    run_mcnemar_significance_test
)
from src.pipeline import build_production_pipeline, validate_pipeline
from src.export_onnx import save_pipeline_joblib, export_to_onnx, verify_onnx_parity
from src.monitoring import generate_psi_report, plot_drift_dashboard
from simulate_production import simulate_production_monitoring


def main():
    print("=" * 80)
    print("END-TO-END DIABETES RISK PREDICTION SYSTEM (CO1 - CO6)")
    print("CLINICAL SCREENING TOOL (NOT A DIAGNOSTIC DEVICE)")
    print("PRIORITIZING RECALL, PRECISION, AND PR-AUC")
    print("=" * 80)

    # -------------------------------------------------------------
    # STAGE 1: Data Ingestion & Summarisation (CO1)
    # -------------------------------------------------------------
    print("\n>>> STAGE 1 [CO1]: Loading Data & Summary Statistics...")
    df_raw = load_raw_data()
    summary = summarise_dataset(df_raw)
    print(f"Loaded {summary['num_rows']} patient records with {summary['num_columns']} columns.")
    print(f"Class Distribution: {summary['outcome_distribution']} (Positive Ratio: {summary['outcome_ratio'][1]*100:.1f}%)")
    print(f"Physiological zero counts: {summary['zero_counts']}")

    df_cleaned = clean_dataset(df_raw)
    X_train, y_train, X_val, y_val, X_test, y_test = get_stratified_splits(df_cleaned)
    print(f"Stratified Split Sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # -------------------------------------------------------------
    # STAGE 2: Exploratory Data Analysis (CO1)
    # -------------------------------------------------------------
    print("\n>>> STAGE 2 [CO1]: Generating Publication-Grade EDA Visualizations...")
    eda_stats = run_eda(df_raw, df_cleaned)
    print(f"Top 3 Correlated Features with Outcome: {list(eda_stats['top_correlations_with_target'].items())[:3]}")

    # -------------------------------------------------------------
    # STAGE 3: Production Pipeline Fit & Validation (CO6 / CO1)
    # -------------------------------------------------------------
    print("\n>>> STAGE 3 [CO6]: Building Atomic Scikit-Learn Pipeline...")
    pipeline = build_production_pipeline()
    pipeline.fit(X_train, y_train)
    validate_pipeline(pipeline, X_train, y_train, X_val)

    # -------------------------------------------------------------
    # STAGE 4: Lifecycle Tracing & Skew Diagnosis (CO1)
    # -------------------------------------------------------------
    print("\n>>> STAGE 4 [CO1]: Tracing Prediction Request & Diagnosing Skew...")
    sample_patient = {
        "Pregnancies": 3,
        "Glucose": 158.0,
        "BloodPressure": 76.0,
        "SkinThickness": 32.0,
        "Insulin": 0.0,  # Zero clinical test -> properly converted to NaN
        "BMI": 34.5,
        "DiabetesPedigreeFunction": 0.58,
        "Age": 48
    }
    trace_res = trace_prediction_request(sample_patient, pipeline, decision_threshold=0.35)
    print(f"Request Trace Completed in {trace_res['total_latency_ms']:.2f} ms")
    print(f"Predicted Risk Score: {trace_res['final_prediction']['risk_score']} | Tier: {trace_res['final_prediction']['risk_tier']}")

    # Simulate Skew
    skewed_sample = X_val.copy().fillna(0)  # Skew: raw zeros passed into model without sanitization
    skew_diag = diagnose_training_serving_skew(X_val, skewed_sample, pipeline)
    print(f"Training-Serving Skew Max Discrepancy: {skew_diag['max_absolute_prediction_discrepancy']:.4f}")

    # -------------------------------------------------------------
    # STAGE 5: Linear Models, Regularisation & Scratch GD (CO2)
    # -------------------------------------------------------------
    print("\n>>> STAGE 5 [CO2]: Benchmarking Linear Models & Scratch Gradient Descent...")
    prep = pipeline.named_steps["preprocessor"]
    X_tr_prep = prep.transform(X_train)
    X_va_prep = prep.transform(X_val)
    X_te_prep = prep.transform(X_test)

    gd_res = run_gradient_descent_benchmark(X_tr_prep, y_train.values, X_va_prep, y_val.values)
    print(f"Batch GD RMSE: {gd_res['batch_gd_val_rmse']:.4f} | Mini-Batch GD RMSE: {gd_res['minibatch_gd_val_rmse']:.4f} | Sklearn OLS: {gd_res['sklearn_ols_val_rmse']:.4f}")

    reg_df, scaling_res = compare_regularisation(pd.DataFrame(X_tr_prep), y_train, pd.DataFrame(X_va_prep), y_val)
    print("Regularization Sparsity (L1 vs L2 non-zeros at C=0.01):")
    c_001 = reg_df.loc[reg_df["C"] == 0.01].iloc[0]
    print(f"  Lasso non-zeros: {c_001['Lasso_nonzero']} | Ridge non-zeros: {c_001['Ridge_nonzero']}")
    print(f"Scaling Impact: Unscaled solver took {scaling_res['unscaled_iterations']} iters vs {scaling_res['scaled_iterations']} iters scaled (Speedup: {scaling_res['iterations_reduction_factor']:.1f}x)")

    # -------------------------------------------------------------
    # STAGE 6: Tree-Based Models, Ensembles, SVM & Explainability (CO3)
    # -------------------------------------------------------------
    print("\n>>> STAGE 6 [CO3]: Tree Depth Sweep, Pruning, Random Forest OOB & Benchmarking...")
    dt_sweep_df, dt_summary = sweep_decision_tree_depth(pd.DataFrame(X_tr_prep), y_train, pd.DataFrame(X_va_prep), y_val)
    print(f"Decision Tree Optimal Depth: {dt_summary['optimal_depth']} (Val Acc: {dt_summary['optimal_depth_val_acc']:.3f})")
    print(f"Full Tree Overfit Train Acc: {dt_summary['full_tree_overfit_train_acc']:.3f} -> Overfitting Gap: {dt_summary['full_tree_overfit_train_acc'] - dt_summary['full_tree_val_acc']:.3f}")

    rf_oob_df, best_rf = benchmark_random_forest_oob(pd.DataFrame(X_tr_prep), y_train, pd.DataFrame(X_va_prep), y_val)
    print(f"Random Forest Best OOB Score: {rf_oob_df['OOB_acc'].max():.3f} vs Single Tree: {rf_oob_df['DT_val_acc'].iloc[0]:.3f}")

    all_models_df, trained_models = benchmark_all_supervised_models(
        pd.DataFrame(X_tr_prep), y_train, pd.DataFrame(X_va_prep), y_val
    )
    print("\nSupervised Models Benchmark (Sorted by Recall):")
    print(all_models_df[["Model", "Recall", "Precision", "ROC-AUC", "Accuracy"]].to_string(index=False))

    # SHAP & Feature Importance
    transformed_feature_names = [f"feat_{i}" for i in range(X_tr_prep.shape[1])]
    for idx, col in enumerate(FEATURE_COLUMNS):
        if idx < len(transformed_feature_names):
            transformed_feature_names[idx] = col
    shap_res = compute_explainability_and_shap(best_rf, pd.DataFrame(X_tr_prep), pd.DataFrame(X_va_prep), y_val, transformed_feature_names)
    print(f"Top Features by SHAP Impact: {list(shap_res['top_features_shap'].keys())[:3]}")

    # -------------------------------------------------------------
    # STAGE 7: Unsupervised Learning & Dimensionality Reduction (CO4)
    # -------------------------------------------------------------
    print("\n>>> STAGE 7 [CO4]: K-Means, Hierarchical, DBSCAN, PCA, t-SNE & Anomaly Detection...")
    km_df, best_km = run_kmeans_sweep(X_tr_prep)
    print(f"K-Means Optimal Silhouette: {km_df['Silhouette'].max():.3f} at K={km_df.loc[km_df['Silhouette'].idxmax()]['K']}")

    run_hierarchical_and_dendrogram(X_tr_prep)
    db, db_stats = run_dbscan_analysis(X_tr_prep)
    print(f"DBSCAN detected {db_stats['noise_points_detected']} clinical outlier / noise points ({db_stats['noise_percentage']:.1f}%)")

    dim_res = run_dimensionality_reduction_dashboard(X_train.fillna(X_train.median()).values, X_tr_prep, y_train.values)
    print(f"PCA Scaled Top-2 Variance: {dim_res['scaled_pca_top2_cumulative']*100:.1f}%")
    print(f"PCA Scaling Insight: {dim_res['scaling_pitfall_note']}")

    iso_res = run_isolation_forest_anomaly_detection(X_tr_prep, y_train.values)
    print(f"Isolation Forest flagged {iso_res['n_anomalies_detected']} anomalous profiles ({iso_res['anomaly_percentage']:.1f}%)")

    # -------------------------------------------------------------
    # STAGE 8: Model Evaluation, Selection & Calibration (CO5)
    # -------------------------------------------------------------
    print("\n>>> STAGE 8 [CO5]: Nested CV, Calibration, Threshold Tuning & Significance...")
    from sklearn.ensemble import RandomForestClassifier
    rf_estimator = RandomForestClassifier(random_state=42)
    nested_res = run_nested_cv_benchmark(
        rf_estimator, X_tr_prep, y_train.values,
        param_grid={"n_estimators": [50, 100], "max_depth": [3, 5]}
    )
    print(f"Nested CV ROC-AUC: {nested_res['nested_cv_mean']} (Optimism Bias: {nested_res['optimism_bias']:.4f})")

    # Test Set Predictions using Best Model
    rf_clf = trained_models["Random Forest"]
    test_probs = rf_clf.predict_proba(X_te_prep)[:, 1]
    default_metrics = compute_clinical_metrics_suite(y_test.values, test_probs, threshold=0.50)
    print(f"\nDefault Test Metrics (Threshold=0.50): Accuracy={default_metrics['accuracy']:.3f}, Precision={default_metrics['precision']:.3f}, Recall={default_metrics['recall_sensitivity']:.3f}, PR-AUC={default_metrics['pr_auc']:.3f}")
    print(f"  False Negatives (Missed Risk): {default_metrics['confusion_matrix']['FN']} of {sum(y_test)} true diabetic cases.")

    # Threshold Tuning targeting high Recall >= 0.90
    thresh_df, rec_thresh = sweep_classification_threshold(y_test.values, test_probs, recall_target=CLINICAL_RECALL_TARGET)
    tuned_metrics = compute_clinical_metrics_suite(y_test.values, test_probs, threshold=rec_thresh)
    print(f"\nTuned Clinical Metrics (Threshold={rec_thresh:.2f}): Recall={tuned_metrics['recall_sensitivity']:.3f}, Precision={tuned_metrics['precision']:.3f}, PR-AUC={tuned_metrics['pr_auc']:.3f}")
    print(f"  False Negatives Reduced to: {tuned_metrics['confusion_matrix']['FN']} (from {default_metrics['confusion_matrix']['FN']})")

    # Probability Calibration
    calibrated_clf, cal_summary = calibrate_and_plot_reliability(rf_clf, pd.DataFrame(X_tr_prep), y_train, pd.DataFrame(X_va_prep), y_val)
    print(f"Platt Calibration Brier Score: {cal_summary['brier_uncalibrated']:.4f} -> {cal_summary['brier_sigmoid_platt']:.4f} ({cal_summary['brier_improvement_percent']}% improvement)")

    # McNemar's Test: Random Forest vs Logistic Regression
    lr_clf = trained_models["Logistic Regression"]
    mcnemar_res = run_mcnemar_significance_test(
        y_test.values,
        rf_clf.predict(X_te_prep),
        lr_clf.predict(X_te_prep),
        name_a="Random Forest",
        name_b="Logistic Regression"
    )
    print(f"McNemar Test Result: {mcnemar_res['interpretation']} (p = {mcnemar_res['p_value']:.4f})")

    # -------------------------------------------------------------
    # STAGE 9: Model Packaging, ONNX Export & Parity Assertion (CO6)
    # -------------------------------------------------------------
    print("\n>>> STAGE 9 [CO6]: Packaging Model to Joblib and ONNX...")
    save_pipeline_joblib(pipeline, MODEL_JOBLIB_PATH)
    export_to_onnx(pipeline.named_steps["classifier"], n_features=X_tr_prep.shape[1], output_path=MODEL_ONNX_PATH)
    parity_res = verify_onnx_parity(pipeline, MODEL_ONNX_PATH, X_test.head(15))
    print(f"ONNX Inference Parity: {parity_res['status']} (Max diff: {parity_res['max_absolute_difference']:.2e})")

    # -------------------------------------------------------------
    # STAGE 10: Production Drift Observability Simulation (CO6)
    # -------------------------------------------------------------
    print("\n>>> STAGE 10 [CO6]: Running Production Monitoring & Drift Simulation...")
    simulate_production_monitoring()

    # -------------------------------------------------------------
    # STAGE 11: Execute Pytest Test Suite
    # -------------------------------------------------------------
    print("\n>>> STAGE 11: Executing Automated Pytest Test Suite...")
    pytest_exit_code = pytest.main(["-v", "tests"])
    assert pytest_exit_code == 0, f"Pytest tests failed with exit code {pytest_exit_code}"

    print("\n" + "=" * 80)
    print("ALL MODULES AND TESTS EXECUTED SUCCESSFULLY WITH ZERO ERRORS!")
    print("ALL VISUAL AND SERIALIZED ARTIFACTS GENERATED IN artifacts/")
    print("=" * 80)


if __name__ == "__main__":
    main()
