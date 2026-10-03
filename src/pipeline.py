"""
CO6: Production Pipeline Module ensuring Zero Training-Serving Skew.
"""
from typing import List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.config import FEATURE_COLUMNS, PHYSIOLOGICAL_ZERO_COLUMNS, RANDOM_STATE


class PhysiologicalZeroSanitizer(BaseEstimator, TransformerMixin):
    """
    Custom transformer converting physiological zero values (Glucose, BloodPressure,
    SkinThickness, Insulin, BMI) to NaN to avoid distorted measurements.
    """
    def __init__(self, zero_columns: List[str] = None):
        self.zero_columns = zero_columns if zero_columns is not None else PHYSIOLOGICAL_ZERO_COLUMNS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_out = X.copy()
            for col in self.zero_columns:
                if col in X_out.columns:
                    X_out[col] = X_out[col].replace(0, np.nan)
            return X_out
        else:
            X_out = np.copy(X)
            # If numpy array, assume columns match FEATURE_COLUMNS order
            for col in self.zero_columns:
                if col in FEATURE_COLUMNS:
                    idx = FEATURE_COLUMNS.index(col)
                    X_out[X_out[:, idx] == 0, idx] = np.nan
            return X_out


def build_preprocessor() -> Pipeline:
    """
    Builds the complete atomic preprocessor pipeline:
    1. Zero Sanitizer (0 -> NaN)
    2. Missing Indicator + Median Imputation
    3. Standard Scaling
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    indicator_pipe = Pipeline([
        ("indicator", MissingIndicator(features="missing-only"))
    ])

    features_union = FeatureUnion([
        ("numeric_scaled", numeric_pipe),
        ("missing_flags", indicator_pipe)
    ])

    preprocessor = Pipeline([
        ("sanitizer", PhysiologicalZeroSanitizer()),
        ("union", features_union)
    ])
    return preprocessor


def build_production_pipeline(classifier=None) -> Pipeline:
    """
    Constructs an end-to-end atomic production pipeline containing both
    preprocessing and classification estimator.
    """
    if classifier is None:
        classifier = RandomForestClassifier(
            n_estimators=150,
            max_features="sqrt",
            random_state=RANDOM_STATE
        )

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier)
    ])
    return pipeline


def validate_pipeline(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame) -> bool:
    """
    Asserts pipeline integrity:
    1. No NaNs after transformation.
    2. Correct transformed shape.
    3. Output probabilities are bounded in [0, 1].
    """
    pipeline.fit(X_train, y_train)
    prep = pipeline.named_steps["preprocessor"]
    X_val_trans = prep.transform(X_val)

    assert not np.isnan(X_val_trans).any(), "Pipeline output contains unexpected NaN values"
    assert X_val_trans.shape[0] == len(X_val), "Transformed rows count mismatch"
    assert X_val_trans.shape[1] >= len(FEATURE_COLUMNS), "Transformed feature count less than input"

    probs = pipeline.predict_proba(X_val)[:, 1]
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0), "Predicted probabilities outside [0, 1]"

    print("Pipeline validation PASSED: zero NaNs, valid dimensions, consistent transform.")
    return True
