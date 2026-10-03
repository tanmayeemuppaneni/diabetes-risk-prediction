"""
Data ingestion, validation, and splitting module.
"""
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    PHYSIOLOGICAL_ZERO_COLUMNS,
    RANDOM_STATE
)


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Pima Indians Diabetes CSV dataset."""
    df = pd.read_csv(path)
    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    return df


def summarise_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Summarise dataset shape, dtypes, missing values (both literal NaN and clinical zeros),
    and class distribution.
    """
    summary = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "outcome_distribution": df[TARGET_COLUMN].value_counts().to_dict(),
        "outcome_ratio": df[TARGET_COLUMN].value_counts(normalize=True).to_dict(),
        "zero_counts": {
            col: int((df[col] == 0).sum())
            for col in PHYSIOLOGICAL_ZERO_COLUMNS
            if col in df.columns
        }
    }
    return summary


def clean_dataset(df: pd.DataFrame, replace_zeros: bool = True) -> pd.DataFrame:
    """
    Process raw data:
    If replace_zeros=True, replace biologically impossible zero values with NaN
    so that downstream pipelines can apply proper imputation without data leakage.
    """
    df_cleaned = df.copy()
    if replace_zeros:
        for col in PHYSIOLOGICAL_ZERO_COLUMNS:
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].replace(0, np.nan)
    return df_cleaned


def get_stratified_splits(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Generate a 3-way 70/15/15 stratified train/val/test split.
    Guarantees strict isolation of validation and test sets.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # From remaining temp (85%), split validation set (0.15 / 0.85 approx 0.1765)
    adjusted_val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=adjusted_val_ratio,
        stratify=y_temp,
        random_state=random_state
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Standard 80/20 train/test split for K-fold CV workflows.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
