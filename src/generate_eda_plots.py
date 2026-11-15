"""EDA Plot Generation Script for CreditGuard AI.

Generates the 10 comprehensive EDA charts required by Section 4
and saves them to outputs/visualizations/.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.utils import load_dataset, VIS_DIR
from src.preprocessing import clean_raw_data

# Set styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
palette = ["#2563EB", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6"]


def generate_all_eda_plots():
    print("Loading cs-training.csv for EDA plot generation...")
    df = load_dataset("cs-training.csv")
    df = clean_raw_data(df)
    print(f"Dataset shape: {df.shape}")

    # 1. Target Class Distribution
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    counts = df["SeriousDlqin2yrs"].value_counts()
    bars = ax.bar(["0: No Delinquency", "1: Serious Delinquency"], counts, color=[palette[0], palette[1]], width=0.55, edgecolor="black")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1500, f"{yval:,} ({yval/len(df):.2%})", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_title("CreditGuard AI: Target Class Distribution (SeriousDlqin2yrs)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Borrowers", fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "01_target_distribution.png", bbox_inches="tight")
    plt.close()
    print("Saved 01_target_distribution.png")

    # 2. Target Percentage Pie Chart
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.pie(
        counts,
        labels=["No Delinquency (0)", "Serious Delinquency (1)"],
        autopct="%1.2f%%",
        startangle=140,
        colors=[palette[0], palette[1]],
        explode=(0, 0.1),
        textprops={"fontweight": "bold", "fontsize": 11},
    )
    ax.set_title("CreditGuard AI: Target Class Proportion (Imbalance Ratio 14:1)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "02_target_percentage.png", bbox_inches="tight")
    plt.close()
    print("Saved 02_target_percentage.png")

    # 3. Age Distribution
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    sns.histplot(data=df, x="age", hue="SeriousDlqin2yrs", bins=40, kde=True, palette=[palette[0], palette[1]], ax=ax, alpha=0.6)
    ax.set_title("CreditGuard AI: Borrower Age Distribution by Delinquency Status", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Borrower Age (Years)", fontweight="bold")
    ax.set_ylabel("Density / Count", fontweight="bold")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "03_age_distribution.png", bbox_inches="tight")
    plt.close()
    print("Saved 03_age_distribution.png")

    # 4. Monthly Income Distribution (Log Scaled)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    valid_income = df["MonthlyIncome"].dropna()
    log_income = np.log1p(valid_income.clip(lower=0))
    sns.histplot(log_income, bins=45, kde=True, color=palette[2], ax=ax, edgecolor="black")
    ax.set_title("CreditGuard AI: Monthly Income Distribution (Natural Log Transformed)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("ln(1 + Monthly Income in $)", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    median_val = np.median(log_income)
    ax.axvline(median_val, color="red", linestyle="--", label=f"Median log income: {median_val:.2f} (~${np.expm1(median_val):,.0f})")
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "04_monthly_income_distribution.png", bbox_inches="tight")
    plt.close()
    print("Saved 04_monthly_income_distribution.png")

    # 5. Debt Ratio Distribution
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    # Clip extreme debt ratios for visualization
    debt_clipped = df["DebtRatio"].clip(upper=5.0)
    sns.histplot(debt_clipped, bins=50, kde=False, color=palette[3], ax=ax, edgecolor="black")
    ax.set_title("CreditGuard AI: Debt Ratio Distribution (Clipped at 5.0 for Viewability)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Debt Ratio (Monthly Debt Obligations / Monthly Gross Income)", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "05_debt_ratio_distribution.png", bbox_inches="tight")
    plt.close()
    print("Saved 05_debt_ratio_distribution.png")

    # 6. Credit Utilization Distribution
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    util_clipped = df["RevolvingUtilizationOfUnsecuredLines"].clip(upper=2.0)
    sns.histplot(data=df.assign(util_clipped=util_clipped), x="util_clipped", hue="SeriousDlqin2yrs", bins=50, palette=[palette[0], palette[1]], ax=ax, alpha=0.6)
    ax.set_title("CreditGuard AI: Revolving Credit Utilization Distribution", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Credit Utilization Ratio (Balance / Limit)", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.axvline(1.0, color="darkred", linestyle=":", label="100% Credit Limit Threshold")
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "06_credit_utilization_distribution.png", bbox_inches="tight")
    plt.close()
    print("Saved 06_credit_utilization_distribution.png")

    # 7. Late Payment Distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
    cols = ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate"]
    titles = ["30-59 Days Past Due", "60-89 Days Past Due", "90+ Days Past Due"]
    for ax, col, title in zip(axes, cols, titles):
        # Filter extreme sentinels 96/98 for clean histogram display
        filtered = df[col].apply(lambda v: 10 if v in [96, 98] else min(v, 10))
        sns.countplot(x=filtered, ax=ax, palette="Blues_r")
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel("Number of Occurrences (Capped at 10)", fontweight="bold")
        ax.set_ylabel("Count", fontweight="bold")
    plt.suptitle("CreditGuard AI: Historical Delinquency Count Distributions", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "07_late_payment_distribution.png", bbox_inches="tight")
    plt.close()
    print("Saved 07_late_payment_distribution.png")

    # 8. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-0.2, vmax=1.0, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("CreditGuard AI: Correlation Heatmap Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "08_correlation_heatmap.png", bbox_inches="tight")
    plt.close()
    print("Saved 08_correlation_heatmap.png")

    # 9. Boxplots for Important Financial Variables
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    box_vars = [
        ("age", "Age", (18, 90)),
        ("NumberOfOpenCreditLinesAndLoans", "Open Credit Lines", (0, 30)),
        ("RevolvingUtilizationOfUnsecuredLines", "Utilization (Clipped)", (0, 2.0)),
    ]
    for ax, (var, label, y_lim) in zip(axes, box_vars):
        temp_data = df.copy()
        if var == "RevolvingUtilizationOfUnsecuredLines":
            temp_data[var] = temp_data[var].clip(upper=2.0)
        sns.boxplot(data=temp_data, x="SeriousDlqin2yrs", y=var, palette=[palette[0], palette[1]], ax=ax, showfliers=False)
        ax.set_title(f"{label} by Delinquency Status", fontweight="bold", fontsize=11)
        ax.set_xticklabels(["0: No Delinquency", "1: Delinquency"], fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(label, fontweight="bold")
        ax.set_ylim(y_lim)
    plt.suptitle("CreditGuard AI: Financial Indicator Distributions by Target Class", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "09_boxplots_financial_variables.png", bbox_inches="tight")
    plt.close()
    print("Saved 09_boxplots_financial_variables.png")

    # 10. Feature Relationship Visualizations (Age vs Income colored by Delinquency)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    sample_df = df.dropna(subset=["MonthlyIncome"]).sample(min(5000, len(df)), random_state=42)
    scatter = ax.scatter(
        sample_df["age"],
        np.log1p(sample_df["MonthlyIncome"].clip(lower=0, upper=50000)),
        c=sample_df["SeriousDlqin2yrs"],
        cmap="coolwarm",
        alpha=0.5,
        s=18,
    )
    cbar = plt.colorbar(scatter, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(["0: Non-Delinquent", "1: Delinquent"])
    ax.set_title("CreditGuard AI: Age vs. ln(Monthly Income) Stratified by Delinquency", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Borrower Age (Years)", fontweight="bold")
    ax.set_ylabel("ln(1 + Monthly Income)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "10_feature_relationships.png", bbox_inches="tight")
    plt.close()
    print("Saved 10_feature_relationships.png")

    print("\nAll 10 EDA visualizations successfully generated and saved to outputs/visualizations/!")


if __name__ == "__main__":
    generate_all_eda_plots()
