"""Utility functions for CreditGuard AI.

Handles path resolutions, data loading, risk categorizations,
and plotting configurations.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
VIS_DIR = OUTPUTS_DIR / "visualizations"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Ensure runtime directories exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, VIS_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Feature metadata and descriptions
RAW_FEATURE_NAMES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

TARGET_COL = "SeriousDlqin2yrs"

FEATURE_DESCRIPTIONS = {
    "RevolvingUtilizationOfUnsecuredLines": "Total balance on credit cards and personal lines divided by credit limit (fraction)",
    "age": "Age of borrower in years",
    "NumberOfTime30-59DaysPastDueNotWorse": "Number of times borrower was 30-59 days past due in last 2 years",
    "DebtRatio": "Monthly debt payments and living costs divided by monthly gross income",
    "MonthlyIncome": "Monthly gross income in currency units",
    "NumberOfOpenCreditLinesAndLoans": "Number of open credit loans and lines of credit",
    "NumberOfTimes90DaysLate": "Number of times borrower was 90+ days past due",
    "NumberRealEstateLoansOrLines": "Number of mortgage and real estate loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "Number of times borrower was 60-89 days past due in last 2 years",
    "NumberOfDependents": "Number of family dependents excluding borrower",
}


def load_dataset(filename="cs-training.csv"):
    """Load a dataset from the data directory and remove redundant index column if present."""
    file_path = DATA_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset '{filename}' not found at {file_path}. "
            f"Please run 'python src/download_data.py' first."
        )

    df = pd.read_csv(file_path)

    # Check for redundant index column (e.g., 'Unnamed: 0')
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    return df


def categorize_risk(probability, low_threshold=0.15, high_threshold=0.35):
    """Assign an illustrative financial distress risk category based on model probability.

    Parameters
    ----------
    probability : float
        Predicted probability of serious financial delinquency (between 0 and 1).
    low_threshold : float
        Threshold below which borrower is categorized as Low Risk (default 0.15).
    high_threshold : float
        Threshold above which borrower is categorized as High Risk (default 0.35).

    Returns
    -------
    dict: category name, color code, and descriptive assessment.
    """
    if probability < low_threshold:
        return {
            "category": "Low Risk",
            "tier": "Tier 1",
            "color": "#10B981",  # Emerald Green
            "badge": "success",
            "description": "Borrower demonstrates strong financial stability with minimal likelihood of delinquency.",
            "recommendation": "Standard terms applicable. Low credit default probability.",
        }
    elif probability <= high_threshold:
        return {
            "category": "Moderate Risk",
            "tier": "Tier 2",
            "color": "#F59E0B",  # Amber / Warning
            "badge": "warning",
            "description": "Borrower exhibits moderate leverage or occasional short-term payment delays.",
            "recommendation": "Secondary documentation or collateral verification recommended.",
        }
    else:
        return {
            "category": "High Risk",
            "tier": "Tier 3",
            "color": "#EF4444",  # Crimson / Danger
            "badge": "danger",
            "description": "Borrower shows elevated financial stress, high credit utilization, or past delinquency.",
            "recommendation": "Enhanced scrutiny or mitigation required. Elevated probability of distress.",
        }
