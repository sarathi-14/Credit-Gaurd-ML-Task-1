"""Data preprocessing and pipeline building module for CreditGuard AI.

Implements leakage-free preprocessing pipelines, feature imputation,
scaling, and stratified train-validation splitting.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from src.utils import TARGET_COL, RAW_FEATURE_NAMES
from src.feature_engineering import CreditFeatureEngineer


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform baseline sanity cleaning on raw input data.

    - Replaces invalid age entries (e.g., age <= 0) with median adult age.
    """
    clean_df = df.copy()

    # In the Give Me Some Credit dataset, exactly 1 row has age = 0
    if "age" in clean_df.columns:
        invalid_age = clean_df["age"] <= 0
        if invalid_age.any():
            median_valid_age = clean_df.loc[clean_df["age"] > 0, "age"].median()
            clean_df.loc[invalid_age, "age"] = median_valid_age

    return clean_df


def split_features_and_target(df: pd.DataFrame):
    """Separate raw features and the binary target variable.

    Returns
    -------
    X : pd.DataFrame
        Input features.
    y : pd.Series or None
        Target Series if TARGET_COL exists, else None.
    """
    df_cleaned = clean_raw_data(df)

    if TARGET_COL in df_cleaned.columns:
        X = df_cleaned[RAW_FEATURE_NAMES].copy()
        y = df_cleaned[TARGET_COL].astype(int).copy()
        return X, y
    else:
        X = df_cleaned[RAW_FEATURE_NAMES].copy()
        return X, None


def get_stratified_train_val_split(X, y, test_size=0.20, random_state=42):
    """Generate a stratified train-validation split to preserve class imbalance ratios.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target vector.
    test_size : float
        Proportion of dataset held out for validation (default 0.20).
    random_state : int
        Seed for reproducibility (default 42).

    Returns
    -------
    X_train, X_val, y_train, y_val
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
        shuffle=True,
    )


def create_preprocessor(use_scaler=False):
    """Create a standalone Scikit-learn preprocessing pipeline.

    Parameters
    ----------
    use_scaler : bool
        Whether to append a RobustScaler/StandardScaler step (recommended for Logistic Regression).
    """
    steps = [
        ("engineer", CreditFeatureEngineer()),
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if use_scaler:
        steps.append(("scaler", RobustScaler()))

    return Pipeline(steps)


def create_model_pipeline(classifier, use_scaler=False):
    """Combine preprocessing and an estimator into an end-to-end reproducible pipeline.

    Parameters
    ----------
    classifier : Scikit-learn estimator
        The classifier instance (e.g. LogisticRegression, DecisionTreeClassifier, RandomForestClassifier).
    use_scaler : bool
        Whether to scale numerical features prior to classification.

    Returns
    -------
    Pipeline
        End-to-end executable Scikit-learn pipeline.
    """
    steps = [
        ("engineer", CreditFeatureEngineer()),
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if use_scaler:
        steps.append(("scaler", RobustScaler()))

    steps.append(("classifier", classifier))

    return Pipeline(steps)
