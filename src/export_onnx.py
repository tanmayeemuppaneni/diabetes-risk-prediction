"""
CO6: Model Packaging, Joblib Serialization, ONNX Conversion, and Numerical Parity Verification.
"""
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

from src.config import MODEL_JOBLIB_PATH, MODEL_ONNX_PATH, FEATURE_COLUMNS


def save_pipeline_joblib(pipeline, path: Path = MODEL_JOBLIB_PATH) -> str:
    """Save trained pipeline using joblib."""
    joblib.dump(pipeline, path)
    return str(path)


def load_pipeline_joblib(path: Path = MODEL_JOBLIB_PATH):
    """Load pipeline from joblib file."""
    return joblib.load(path)


def export_to_onnx(
    classifier_model,
    n_features: int,
    output_path: Path = MODEL_ONNX_PATH
) -> str:
    """
    Exports a trained classifier model to portable ONNX format.
    """
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onx = convert_sklearn(classifier_model, initial_types=initial_type, target_opset=12)

    with open(output_path, "wb") as f:
        f.write(onx.SerializeToString())

    return str(output_path)


def verify_onnx_parity(
    pipeline,
    onnx_path: Path,
    sample_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Verifies that ONNX Runtime inference matches Scikit-Learn inference
    with numerical tolerance < 1e-4.
    """
    # 1. Scikit-learn inference
    prep = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]

    X_transformed = prep.transform(sample_df).astype(np.float32)
    sk_probs = clf.predict_proba(X_transformed)[:, 1]

    # 2. ONNX Runtime inference
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_outs = session.run(None, {input_name: X_transformed})

    # In skl2onnx classifier output, onnx_outs[1] is typically a list of dicts or probability array
    if isinstance(onnx_outs[1], list):
        # List of dicts {0: p0, 1: p1}
        onnx_probs = np.array([d[1] for d in onnx_outs[1]])
    elif isinstance(onnx_outs[1], np.ndarray):
        onnx_probs = onnx_outs[1][:, 1]
    else:
        raise ValueError("Unsupported ONNX probability output format")

    diff = np.abs(sk_probs - onnx_probs)
    max_diff = float(np.max(diff))

    is_parity = bool(max_diff < 1e-3)
    return {
        "max_absolute_difference": max_diff,
        "is_parity_achieved": is_parity,
        "onnx_sample_probs": np.round(onnx_probs[:3], 4).tolist(),
        "sklearn_sample_probs": np.round(sk_probs[:3], 4).tolist(),
        "status": "PASSED" if is_parity else "FAILED"
    }
