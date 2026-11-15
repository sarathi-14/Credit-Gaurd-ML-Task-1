"""Feature engineering module for CreditGuard AI.

Creates domain-specific financial risk indicators and behavioral metrics
using a Scikit-learn compatible Transformer to prevent data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """Transformer that constructs financially interpretable features from raw borrower attributes.

    Engineered features:
    1. TotalPastDue: Aggregated count of past-due payment events across all delinquency buckets.
    2. HasSevereDelinquency: Binary flag for 90+ days past due occurrence.
    3. HasAnyLatePayment: Binary flag indicating any late payment history.
    4. IncomePerPerson: Estimated disposable income per household member.
    5. HighUtilizationFlag: Indicator if revolving credit exceeds 100% of available credit limit.
    6. NoOpenCreditLines: Indicator if the borrower has zero active credit lines.
    7. CleanLatePayments: Replaces synthetic/unknown code markers (96, 98) with capped realistic bounds.
    8. MissingIncomeFlag: Explicit binary indicator capturing that MonthlyIncome was unobserved.
    """

    def __init__(self, clip_utilization=10.0, max_late_cap=10):
        self.clip_utilization = clip_utilization
        self.max_late_cap = max_late_cap
        self.income_median_ = None
        self.dependents_median_ = None
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        """Compute training set medians for leakage-free feature computation."""
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        if "MonthlyIncome" in df.columns:
            self.income_median_ = float(df["MonthlyIncome"].median())
        else:
            self.income_median_ = 5400.0  # Dataset global median fallback

        if "NumberOfDependents" in df.columns:
            self.dependents_median_ = float(df["NumberOfDependents"].median())
        else:
            self.dependents_median_ = 0.0

        return self

    def transform(self, X):
        """Construct engineered features on the provided dataset."""
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            # Fallback if passed as numpy array
            from src.utils import RAW_FEATURE_NAMES
            df = pd.DataFrame(X, columns=RAW_FEATURE_NAMES)

        # 1. Missing value indicators
        missing_income = df["MonthlyIncome"].isna().astype(float)

        # 2. Impute internal working copies for feature ratios
        working_income = df["MonthlyIncome"].fillna(self.income_median_).clip(lower=0)
        working_dependents = df["NumberOfDependents"].fillna(self.dependents_median_).clip(lower=0)

        # 3. Clean extreme / encoded late payment sentinel values (96, 98)
        # In the Kaggle dataset, 96 and 98 are categorical codes for 'missing/unreported'.
        late_30_59 = df["NumberOfTime30-59DaysPastDueNotWorse"].apply(
            lambda v: self.max_late_cap if v in [96, 98] else min(v, self.max_late_cap)
        )
        late_60_89 = df["NumberOfTime60-89DaysPastDueNotWorse"].apply(
            lambda v: self.max_late_cap if v in [96, 98] else min(v, self.max_late_cap)
        )
        late_90_plus = df["NumberOfTimes90DaysLate"].apply(
            lambda v: self.max_late_cap if v in [96, 98] else min(v, self.max_late_cap)
        )

        # 4. Aggregated delinquency counts
        total_past_due = late_30_59 + late_60_89 + late_90_plus
        has_severe_delinquency = (late_90_plus > 0).astype(float)
        has_any_late = (total_past_due > 0).astype(float)

        # 5. Financial ratios
        income_per_person = working_income / (working_dependents + 1.0)
        high_utilization = (df["RevolvingUtilizationOfUnsecuredLines"] > 1.0).astype(float)
        no_open_lines = (df["NumberOfOpenCreditLinesAndLoans"] == 0).astype(float)

        # 6. Clipped utilization to suppress wild outliers while retaining signal
        clipped_utilization = df["RevolvingUtilizationOfUnsecuredLines"].clip(
            lower=0.0, upper=self.clip_utilization
        )

        # Log transformations for heavily skewed positive variables
        log_income = np.log1p(working_income)
        log_debt_ratio = np.log1p(df["DebtRatio"].clip(lower=0, upper=100000.0))

        # Assemble transformed DataFrame
        result_df = pd.DataFrame(
            {
                # Cleaned / Transformed Base Features
                "RevolvingUtilization_Clipped": clipped_utilization,
                "age": df["age"],
                "DebtRatio_Log": log_debt_ratio,
                "MonthlyIncome_Log": log_income,
                "NumberOfOpenCreditLinesAndLoans": df["NumberOfOpenCreditLinesAndLoans"],
                "NumberRealEstateLoansOrLines": df["NumberRealEstateLoansOrLines"],
                "NumberOfDependents": working_dependents,
                # Cleaned Late Payment Counts
                "NumberOfTime30_59_Clean": late_30_59,
                "NumberOfTime60_89_Clean": late_60_89,
                "NumberOfTimes90_Clean": late_90_plus,
                # Engineered Risk Features
                "TotalPastDue": total_past_due,
                "HasSevereDelinquency": has_severe_delinquency,
                "HasAnyLatePayment": has_any_late,
                "IncomePerPerson": income_per_person,
                "HighUtilizationFlag": high_utilization,
                "NoOpenCreditLines": no_open_lines,
                "MissingIncomeFlag": missing_income,
            },
            index=df.index,
        )

        self.feature_names_out_ = list(result_df.columns)
        return result_df

    def get_feature_names_out(self, input_features=None):
        """Return list of engineered feature names."""
        if self.feature_names_out_ is not None:
            return np.array(self.feature_names_out_, dtype=object)
        return np.array(
            [
                "RevolvingUtilization_Clipped",
                "age",
                "DebtRatio_Log",
                "MonthlyIncome_Log",
                "NumberOfOpenCreditLinesAndLoans",
                "NumberRealEstateLoansOrLines",
                "NumberOfDependents",
                "NumberOfTime30_59_Clean",
                "NumberOfTime60_89_Clean",
                "NumberOfTimes90_Clean",
                "TotalPastDue",
                "HasSevereDelinquency",
                "HasAnyLatePayment",
                "IncomePerPerson",
                "HighUtilizationFlag",
                "NoOpenCreditLines",
                "MissingIncomeFlag",
            ],
            dtype=object,
        )
