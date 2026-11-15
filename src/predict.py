"""Inference and batch prediction module for CreditGuard AI.

Provides single-borrower risk prediction with financial explanations,
and generates Kaggle competition test predictions.
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
import joblib

from src.utils import (
    MODELS_DIR,
    OUTPUTS_DIR,
    RAW_FEATURE_NAMES,
    load_dataset,
    categorize_risk,
)


def load_model(model_path=None):
    """Load the persisted champion model pipeline."""
    if model_path is None:
        model_path = MODELS_DIR / "best_credit_model.joblib"

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            f"Please run 'python src/train_model.py' to train and save the model."
        )

    return joblib.load(model_path)


def predict_single_borrower(borrower_data: dict, model=None, threshold=0.5, low_threshold=0.15, high_threshold=0.35):
    """Generate risk assessment and financial distress probability for a single borrower.

    Parameters
    ----------
    borrower_data : dict
        Dictionary containing values for all 10 raw features.
    model : Pipeline, optional
        Fitted Scikit-learn model pipeline. If None, loaded from disk.
    threshold : float
        Binary classification threshold.
    low_threshold : float
        Boundary for Low vs Moderate risk.
    high_threshold : float
        Boundary for Moderate vs High risk.

    Returns
    -------
    dict: Comprehensive prediction outcome with risk categorization and behavioral insights.
    """
    if model is None:
        model = load_model()

    # Create 1-row DataFrame with strict column order
    df_input = pd.DataFrame([borrower_data])[RAW_FEATURE_NAMES]

    # Predict probability of positive class (financial distress)
    proba = float(model.predict_proba(df_input)[0, 1])
    predicted_class = int(proba >= threshold)

    risk_info = categorize_risk(proba, low_threshold=low_threshold, high_threshold=high_threshold)

    # Identify notable behavioral risk factors
    risk_factors = []
    protective_factors = []

    utilization = float(borrower_data.get("RevolvingUtilizationOfUnsecuredLines", 0))
    if utilization > 0.8:
        risk_factors.append(f"High credit utilization ({utilization:.1%}). Balances near or above card limits.")
    elif utilization < 0.3:
        protective_factors.append(f"Prudent credit utilization ({utilization:.1%}). Ample credit headroom.")

    late_30 = int(borrower_data.get("NumberOfTime30-59DaysPastDueNotWorse", 0))
    late_60 = int(borrower_data.get("NumberOfTime60-89DaysPastDueNotWorse", 0))
    late_90 = int(borrower_data.get("NumberOfTimes90DaysLate", 0))

    if late_90 > 0:
        risk_factors.append(f"Severe delinquency record: {late_90} instance(s) of 90+ days past due.")
    if late_30 + late_60 > 0:
        risk_factors.append(f"Short/medium-term delinquency: {late_30 + late_60} instance(s) past due.")
    if late_30 + late_60 + late_90 == 0:
        protective_factors.append("Flawless repayment history: Zero past-due instances in the last 2 years.")

    debt_ratio = float(borrower_data.get("DebtRatio", 0))
    if debt_ratio > 0.5:
        risk_factors.append(f"Elevated debt ratio ({debt_ratio:.2f}). Significant income allocated to obligations.")
    else:
        protective_factors.append(f"Healthy debt-to-income ratio ({debt_ratio:.2f}).")

    age = int(borrower_data.get("age", 45))
    if age < 25:
        risk_factors.append("Limited credit history length (borrower under age 25).")
    elif age >= 40:
        protective_factors.append(f"Established credit profile (borrower age {age}).")

    return {
        "predicted_class": predicted_class,
        "class_label": "High Delinquency Risk (1)" if predicted_class == 1 else "Low Delinquency Risk (0)",
        "probability": round(proba, 4),
        "probability_percentage": f"{proba * 100:.2f}%",
        "risk_category": risk_info["category"],
        "risk_tier": risk_info["tier"],
        "risk_color": risk_info["color"],
        "risk_badge": risk_info["badge"],
        "risk_description": risk_info["description"],
        "recommendation": risk_info["recommendation"],
        "decision_threshold": threshold,
        "risk_factors": risk_factors if risk_factors else ["No major negative risk anomalies detected."],
        "protective_factors": protective_factors if protective_factors else ["Standard profile metrics."],
    }


def generate_kaggle_test_predictions(test_filename="cs-test.csv", model=None):
    """Generate probability predictions on the unlabeled Kaggle test dataset."""
    print("=" * 70)
    print("CREDITGUARD AI: KAGGLE TEST PREDICTION GENERATOR")
    print("=" * 70)

    if model is None:
        model = load_model()

    test_path = Path(__file__).resolve().parent.parent / "data" / test_filename
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found at: {test_path}")

    print(f"Loading {test_filename}...")
    df_raw = pd.read_csv(test_path)
    print(f"Test rows: {len(df_raw):,}")

    # Kaggle test sets use the original index column as 'Id'
    if "Unnamed: 0" in df_raw.columns:
        test_ids = df_raw["Unnamed: 0"].values
    else:
        test_ids = np.arange(1, len(df_raw) + 1)

    X_test = df_raw[RAW_FEATURE_NAMES].copy()

    print("Generating positive-class probability predictions...")
    probabilities = model.predict_proba(X_test)[:, 1]

    submission_df = pd.DataFrame(
        {
            "Id": test_ids,
            "SeriousDlqin2yrs": probabilities,
        }
    )

    out_path = OUTPUTS_DIR / "credit_scoring_predictions.csv"
    submission_df.to_csv(out_path, index=False)
    print(f"Successfully generated Kaggle predictions at: {out_path}")
    print(f"Sample predictions (first 5):\n{submission_df.head()}")

    return submission_df


if __name__ == "__main__":
    import json

    # Test sample borrower
    sample_borrower = {
        "RevolvingUtilizationOfUnsecuredLines": 0.32,
        "age": 42,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.28,
        "MonthlyIncome": 6500.0,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2,
    }

    print("Testing single borrower prediction...")
    res = predict_single_borrower(sample_borrower)
    print(json.dumps(res, indent=2))
