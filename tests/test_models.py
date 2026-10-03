"""
Unit tests for supervised models, scratch gradient descent, pruning, and ONNX parity.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import load_raw_data, clean_dataset, get_stratified_splits
from src.linear_models import ScratchLinearRegressionGD
from src.tree_models import sweep_decision_tree_depth, benchmark_random_forest_oob
from src.pipeline import build_production_pipeline
from src.export_onnx import export_to_onnx, verify_onnx_parity
from src.config import ARTIFACTS_DIR


@pytest.fixture(scope="module")
def dataset_splits():
    df_raw = load_raw_data()
    df_clean = clean_dataset(df_raw)
    return get_stratified_splits(df_clean)


def test_scratch_gradient_descent_convergence():
    # Simple synthetic linear regression test
    np.random.seed(42)
    X = np.random.randn(100, 3)
    true_w = np.array([2.0, -1.5, 0.5])
    y = np.dot(X, true_w) + 1.2 + np.random.normal(0, 0.01, size=100)

    gd = ScratchLinearRegressionGD(learning_rate=0.05, epochs=500, batch_size=None)
    gd.fit(X, y)

    # Loss must be monotonically decreasing or substantially lower than initial
    assert gd.loss_history[-1] < gd.loss_history[0] * 0.05
    assert np.allclose(gd.weights, true_w, atol=0.1)


def test_decision_tree_depth_sweep(dataset_splits):
    X_train, y_train, X_val, y_val, _, _ = dataset_splits
    prep = build_production_pipeline().named_steps["preprocessor"]
    X_tr_trans = pd.DataFrame(prep.fit_transform(X_train))
    X_va_trans = pd.DataFrame(prep.transform(X_val))

    df_sweep, summary = sweep_decision_tree_depth(X_tr_trans, y_train, X_va_trans, y_val, max_depths=[1, 3, 5, 8])
    assert "optimal_depth" in summary
    assert summary["full_tree_overfit_train_acc"] >= summary["optimal_depth_val_acc"]


def test_random_forest_oob_evaluation(dataset_splits):
    X_train, y_train, X_val, y_val, _, _ = dataset_splits
    prep = build_production_pipeline().named_steps["preprocessor"]
    X_tr_trans = pd.DataFrame(prep.fit_transform(X_train))
    X_va_trans = pd.DataFrame(prep.transform(X_val))

    df_rf, best_rf = benchmark_random_forest_oob(X_tr_trans, y_train, X_va_trans, y_val, n_estimators_list=[20, 50])
    assert not df_rf.empty
    assert "OOB_acc" in df_rf.columns
    assert df_rf["OOB_acc"].iloc[-1] > 0.60


def test_onnx_export_and_inference_parity(dataset_splits):
    X_train, y_train, X_val, y_val, _, _ = dataset_splits
    pipeline = build_production_pipeline()
    pipeline.fit(X_train, y_train)

    prep = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]

    X_train_trans = prep.transform(X_train).astype(np.float32)
    onnx_file = ARTIFACTS_DIR / "test_model.onnx"
    export_to_onnx(clf, n_features=X_train_trans.shape[1], output_path=onnx_file)

    parity_res = verify_onnx_parity(pipeline, onnx_file, X_val.head(10))
    assert parity_res["is_parity_achieved"], f"ONNX parity failed with max diff {parity_res['max_absolute_difference']}"
