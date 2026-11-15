"""Script to generate the complete, production-grade Jupyter Notebook:
notebooks/Credit_Scoring_Model.ipynb
"""

import nbformat as nbf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
nb = nbf.v4.new_notebook()

cells = []

# ==============================================================================
# SECTION 1: PROJECT INTRODUCTION
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# CreditGuard AI: Intelligent Credit Scoring & Financial Risk Prediction
## CodeAlpha Machine Learning Internship — Task 1: Credit Scoring Model
**Author:** Machine Learning Intern  
**Domain:** Financial Analytics, Credit Risk & Predictive Modeling  
**Problem Type:** Supervised Machine Learning (Binary Classification)  
**Target Variable:** `SeriousDlqin2yrs` (0 = No Delinquency, 1 = Serious Delinquency within 2 Years)  

---

### 1. Project Overview & Background
In the banking and financial services industry, **Credit Scoring** is the statistical evaluation of a borrower's creditworthiness. When individuals or businesses apply for financial credit (such as credit cards, mortgages, or personal loans), lenders must assess the likelihood that the borrower will experience financial distress and fail to meet their debt obligations.

#### Key Terminology:
- **Credit Risk:** The potential monetary loss a lender faces if a borrower defaults on a loan or credit obligation.
- **Financial Delinquency:** A condition where a borrower fails to make a scheduled loan or credit card payment by the contractually mandated due date. In this benchmark project, *serious delinquency* is defined as being **90 days past due or worse** within the next two years.
- **Why Machine Learning?** Traditional credit scoring systems (such as legacy scorecards) rely on static, linear threshold rules. Machine Learning empowers financial institutions to capture subtle non-linear interactions, evaluate multi-dimensional risk profiles, handle missing records adaptively, and dynamically calibrate risk probabilities without human bias.

#### Educational & Analytical Notice:
> **Disclaimer:** This project is developed strictly for educational, analytical, and portfolio assessment purposes. It does not constitute a certified banking credit underwriting engine or automated lending approval system.
"""))

# ==============================================================================
# SECTION 2: IMPORT LIBRARIES & ENVIRONMENT SETUP
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## Section 2: Library Imports & Environment Verification
In this section, we import core Python libraries for data processing, statistical computing, pipeline construction, machine learning modeling, and visualization. We also display the runtime package versions to ensure full reproducibility.
"""))

cells.append(nbf.v4.new_code_cell("""# Data manipulation and numerical operations
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn tools and algorithms
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)

# Serialization
import joblib

# Set visualization aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_palette('muted')

print("--- Installed Environment & Package Versions ---")
print(f"Python Version      : {sys.version.split()[0]}")
print(f"Pandas Version      : {pd.__version__}")
print(f"NumPy Version       : {np.__version__}")
print(f"Scikit-Learn Version: {sklearn.__version__}")
print(f"Joblib Version      : {joblib.__version__}")
"""))

# ==============================================================================
# SECTION 3: DATASET LOADING
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## Section 3: Dataset Loading & Verification
We load the original Kaggle **Give Me Some Credit** training dataset (`cs-training.csv`).  
We inspect its physical dimensions, examine the first and last five rows, verify column data types, check for redundant index columns, and review preliminary summary statistics.
"""))

cells.append(nbf.v4.new_code_cell("""# Define project paths relative to notebook location
NOTEBOOK_DIR = Path.cwd()
PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == 'notebooks' else NOTEBOOK_DIR
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
MODELS_DIR = PROJECT_ROOT / 'models'

train_file = DATA_DIR / 'cs-training.csv'
print(f"Verifying dataset presence at: {train_file}")
assert train_file.exists(), f"Dataset missing! Please place cs-training.csv in {DATA_DIR}"

# Load raw training data
df_raw = pd.read_csv(train_file)
print(f"Successfully loaded dataset with shape: {df_raw.shape[0]:,} rows and {df_raw.shape[1]} columns.\\n")

# Check for redundant index column
if 'Unnamed: 0' in df_raw.columns:
    print("Found redundant row index column 'Unnamed: 0'. Removing it from dataset...")
    df_raw = df_raw.drop(columns=['Unnamed: 0'])
    print(f"Updated shape after removing redundant index: {df_raw.shape}")

# Display first 5 records
print("\\n--- First 5 Records ---")
display(df_raw.head())
"""))

cells.append(nbf.v4.new_code_cell("""# Dataset Structure and Data Types
print("--- Dataset Information & Null Value Summary ---")
df_raw.info()
"""))

cells.append(nbf.v4.new_code_cell("""# Statistical Summary of Financial Variables
print("--- Five-Number Statistical Summary ---")
display(df_raw.describe().T)
"""))

# ==============================================================================
# SECTION 4: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## Section 4: Exploratory Data Analysis (EDA)
Exploratory Data Analysis allows us to uncover data quality anomalies, examine skewed distributions, investigate multicollinearity, and quantify class imbalance.

### Key Aspects Analyzed:
1. **Target Distribution (`SeriousDlqin2yrs`):** Imbalanced classification challenge (~93.3% safe vs ~6.7% delinquent).
2. **Missing Values:** Identifying missingness in `MonthlyIncome` (~19.8%) and `NumberOfDependents` (~2.6%).
3. **Distribution of Financial Ratios:** `RevolvingUtilizationOfUnsecuredLines` and `DebtRatio` contain heavy right-skewed tails.
4. **Delinquency Counts:** `NumberOfTime30-59DaysPastDueNotWorse`, `NumberOfTime60-89DaysPastDueNotWorse`, and `NumberOfTimes90DaysLate` contain special sentinel codes (96, 98).
5. **Correlation Matrix:** Assessing collinearity across delinquency variables.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Target Class Imbalance Analysis
target_counts = df_raw['SeriousDlqin2yrs'].value_counts()
target_pct = df_raw['SeriousDlqin2yrs'].value_counts(normalize=True) * 100

print("--- Target Class Distribution ---")
for cls, count in target_counts.items():
    print(f"Class {cls} ({'Delinquent' if cls == 1 else 'Non-Delinquent'}): {count:,} ({target_pct[cls]:.2f}%)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Bar Chart
sns.barplot(x=target_counts.index, y=target_counts.values, ax=ax1, palette=['#2563EB', '#EF4444'])
ax1.set_title("Target Class Counts (SeriousDlqin2yrs)", fontsize=12, fontweight='bold')
ax1.set_xticklabels(['No Delinquency (0)', 'Serious Delinquency (1)'])
ax1.set_ylabel("Number of Borrowers")

# Pie Chart
ax2.pie(target_counts, labels=['No Delinquency (93.3%)', 'Serious Delinquency (6.7%)'],
        autopct='%1.2f%%', startangle=140, colors=['#2563EB', '#EF4444'], explode=(0, 0.1))
ax2.set_title("Severe Class Imbalance (~14 : 1 Ratio)", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# 2. Missing Value Analysis
missing_counts = df_raw.isnull().sum()
missing_pct = (missing_counts / len(df_raw)) * 100
missing_df = pd.DataFrame({'Missing Records': missing_counts, 'Percentage (%)': missing_pct})
missing_df = missing_df[missing_df['Missing Records'] > 0].sort_values(by='Percentage (%)', ascending=False)

print("--- Missing Values in Dataset ---")
display(missing_df)

plt.figure(figsize=(7, 3.5))
sns.barplot(x=missing_df['Percentage (%)'], y=missing_df.index, palette='Blues_r')
plt.title("Features with Missing Values (%)", fontsize=12, fontweight='bold')
plt.xlabel("Percentage Missing")
plt.xlim(0, 25)
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# 3. Correlation Heatmap
plt.figure(figsize=(10, 7))
corr = df_raw.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-0.2, vmax=1.0)
plt.title("Correlation Matrix of Financial & Demographic Features", fontsize=12, fontweight='bold', pad=12)
plt.show()
"""))

# ==============================================================================
# SECTION 5 & 6: DATA CLEANING & FEATURE ENGINEERING
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## Section 5 & 6: Data Cleaning & Feature Engineering
Raw credit datasets frequently contain extreme outliers, sentinel missing codes, and anomalous records:

1. **Invalid Age Treatment:** The raw dataset contains exactly 1 borrower with `age = 0`. Since an infant cannot hold debt, this value is replaced with the adult median age.
2. **Sentinel Coding in Late Payments:** The values **96** and **98** in the past-due columns represent categorical 'missing/unreported' codes rather than 98 defaults. We cap these at 10 to prevent distorting linear weights and distance metrics.
3. **Engineered Financial Indicators:**
   - `TotalPastDue`: Sum of 30-59, 60-89, and 90+ days past-due occurrences.
   - `HasSevereDelinquency`: Binary indicator of 90+ days past due.
   - `IncomePerPerson`: Estimated household disposable income = `MonthlyIncome / (NumberOfDependents + 1)`.
   - `HighUtilizationFlag`: Binary indicator if revolving credit exceeds 100% of the limit.
   - `MissingIncomeFlag`: Explicit binary indicator signaling that income was not reported.
   - `Log Transformations`: `ln(1 + MonthlyIncome)` and `ln(1 + DebtRatio)` to compress massive skewness.

To ensure **zero data leakage**, all imputations and transformations are encapsulated in a reusable Scikit-learn Transformer.
"""))

cells.append(nbf.v4.new_code_cell("""from sklearn.base import BaseEstimator, TransformerMixin

class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    \"\"\"Custom Scikit-Learn Transformer for leak-free financial feature engineering.\"\"\"
    def __init__(self, clip_utilization=10.0, max_late_cap=10):
        self.clip_utilization = clip_utilization
        self.max_late_cap = max_late_cap
        self.income_median_ = None
        self.dependents_median_ = None

    def fit(self, X, y=None):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.income_median_ = float(df['MonthlyIncome'].median()) if 'MonthlyIncome' in df.columns else 5400.0
        self.dependents_median_ = float(df['NumberOfDependents'].median()) if 'NumberOfDependents' in df.columns else 0.0
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        
        # Missing indicator
        missing_income = df['MonthlyIncome'].isna().astype(float)
        
        # Working imputations
        working_income = df['MonthlyIncome'].fillna(self.income_median_).clip(lower=0)
        working_dependents = df['NumberOfDependents'].fillna(self.dependents_median_).clip(lower=0)
        
        # Clean sentinel codes (96, 98) in late payment attributes
        late_30 = df['NumberOfTime30-59DaysPastDueNotWorse'].apply(lambda v: self.max_late_cap if v in [96, 98] else min(v, self.max_late_cap))
        late_60 = df['NumberOfTime60-89DaysPastDueNotWorse'].apply(lambda v: self.max_late_cap if v in [96, 98] else min(v, self.max_late_cap))
        late_90 = df['NumberOfTimes90DaysLate'].apply(lambda v: self.max_late_cap if v in [96, 98] else min(v, self.max_late_cap))
        
        total_past_due = late_30 + late_60 + late_90
        has_severe_delinq = (late_90 > 0).astype(float)
        has_any_late = (total_past_due > 0).astype(float)
        
        income_per_person = working_income / (working_dependents + 1.0)
        high_util = (df['RevolvingUtilizationOfUnsecuredLines'] > 1.0).astype(float)
        no_open_lines = (df['NumberOfOpenCreditLinesAndLoans'] == 0).astype(float)
        clipped_util = df['RevolvingUtilizationOfUnsecuredLines'].clip(lower=0.0, upper=self.clip_utilization)
        
        log_income = np.log1p(working_income)
        log_debt_ratio = np.log1p(df['DebtRatio'].clip(lower=0, upper=100000.0))
        
        res = pd.DataFrame({
            'RevolvingUtilization_Clipped': clipped_util,
            'age': df['age'],
            'DebtRatio_Log': log_debt_ratio,
            'MonthlyIncome_Log': log_income,
            'NumberOfOpenCreditLinesAndLoans': df['NumberOfOpenCreditLinesAndLoans'],
            'NumberRealEstateLoansOrLines': df['NumberRealEstateLoansOrLines'],
            'NumberOfDependents': working_dependents,
            'NumberOfTime30_59_Clean': late_30,
            'NumberOfTime60_89_Clean': late_60,
            'NumberOfTimes90_Clean': late_90,
            'TotalPastDue': total_past_due,
            'HasSevereDelinquency': has_severe_delinq,
            'HasAnyLatePayment': has_any_late,
            'IncomePerPerson': income_per_person,
            'HighUtilizationFlag': high_util,
            'NoOpenCreditLines': no_open_lines,
            'MissingIncomeFlag': missing_income
        }, index=df.index)
        
        return res

print("CreditFeatureEngineer defined successfully.")
"""))

# ==============================================================================
# SECTION 7: FEATURE & TARGET SEPARATION & STRATIFIED SPLIT
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## Section 7: Feature-Target Separation & Stratified Split
We separate features $X$ from target $y$ (`SeriousDlqin2yrs`).  
Because only ~6.7% of borrowers belong to the delinquent class, a random split risks sampling variability. We apply **Stratified Splitting** (`test_size=0.20, random_state=42, stratify=y`) to guarantee that both the 80% training set and 20% validation set contain the exact same 93.3% / 6.7% class proportion.
"""))

cells.append(nbf.v4.new_code_cell("""# Sanity cleaning on age
df_clean = df_raw.copy()
if (df_clean['age'] <= 0).any():
    med_age = df_clean.loc[df_clean['age'] > 0, 'age'].median()
    df_clean.loc[df_clean['age'] <= 0, 'age'] = med_age

# Split features and target
X = df_clean.drop(columns=['SeriousDlqin2yrs'])
y = df_clean['SeriousDlqin2yrs'].astype(int)

# Stratified Split (80% Train, 20% Validation)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y, shuffle=True
)

print(f"Total Records: {len(X):,}")
print(f"Training Split  : {len(X_train):,} records (Delinquent: {(y_train == 1).sum():,}, {y_train.mean():.2%})")
print(f"Validation Split: {len(X_val):,} records (Delinquent: {(y_val == 1).sum():,}, {y_val.mean():.2%})")
"""))

# ==============================================================================
# SECTION 8 & 9: MACHINE LEARNING ALGORITHMS & PIPELINE DESIGN
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell(r"""## Section 8 & 9: Machine Learning Classification Algorithms
We implement three distinct classification algorithms to compare linear vs. non-linear decision architectures:

### 1. Model 1: Logistic Regression (Regularized Baseline)
- **Principle:** Models the log-odds of delinquency as a linear combination of input features passed through the Sigmoid function: $\sigma(z) = \frac{1}{1 + e^{-z}}$.
- **Handling Imbalance:** Utilizes `class_weight='balanced'` which assigns higher loss penalty inversely proportional to class frequencies.
- **Scaling:** Uses `RobustScaler` to handle extreme financial outliers.

### 2. Model 2: Decision Tree Classifier (Non-linear Tree)
- **Principle:** Recursively partitions the feature space into orthogonal hyper-rectangles using Gini impurity minimization.
- **Regularization:** Constrained with `max_depth=6`, `min_samples_split=25`, and `min_samples_leaf=15` to prevent memorization and leaf overfitting.

### 3. Model 3: Random Forest Classifier (Ensemble Architecture)
- **Principle:** An ensemble of de-correlated decision trees trained via bootstrap aggregation (bagging) with random feature subspace sampling.
- **Parameters:** 100 estimators, `max_depth=10`, `min_samples_split=20`, `min_samples_leaf=10`, and `class_weight='balanced'`.
"""))

cells.append(nbf.v4.new_code_cell("""# Construct end-to-end pipelines for each algorithm

lr_pipeline = Pipeline([
    ('engineer', CreditFeatureEngineer()),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler()),
    ('classifier', LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=42))
])

dt_pipeline = Pipeline([
    ('engineer', CreditFeatureEngineer()),
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', DecisionTreeClassifier(max_depth=6, min_samples_split=25, min_samples_leaf=15, class_weight='balanced', random_state=42))
])

rf_pipeline = Pipeline([
    ('engineer', CreditFeatureEngineer()),
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=20, min_samples_leaf=10, class_weight='balanced', n_jobs=-1, random_state=42))
])

model_pipelines = {
    'Logistic Regression': lr_pipeline,
    'Decision Tree': dt_pipeline,
    'Random Forest': rf_pipeline
}

print("Pipelines successfully configured.")
"""))

# ==============================================================================
# SECTION 10, 11 & 12: MODEL TRAINING, EVALUATION & COMPARISON
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## Section 10, 11 & 12: Model Training, Evaluation & Comparison
We fit each model strictly on `(X_train, y_train)` and evaluate on the unseen validation partition `(X_val, y_val)`.

### Metrics Analyzed:
- **ROC-AUC (Receiver Operating Characteristic Area Under Curve):** Primary metric of the Kaggle competition. Measures discriminative ability across all possible thresholds.
- **PR-AUC (Precision-Recall Area Under Curve):** Critical for imbalanced datasets where the positive class is rare.
- **Recall (Sensitivity):** Proportion of actual delinquent borrowers detected. High recall minimizes costly credit defaults (False Negatives).
- **Precision:** Proportion of flagged borrowers who genuinely default.
- **F1-Score:** Harmonic mean of precision and recall.
"""))

cells.append(nbf.v4.new_code_cell("""import time

results_list = []
eval_dicts = {}

for name, pipeline in model_pipelines.items():
    print(f"Training {name} on 120,000 records...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    t_elapsed = time.time() - t0
    
    # Predict probabilities and classes on held-out validation set
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_val, y_proba)
    pr_auc = average_precision_score(y_val, y_proba)
    
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    prec_curve, rec_curve, _ = precision_recall_curve(y_val, y_proba)
    cm = confusion_matrix(y_val, y_pred)
    
    results_list.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(roc_auc, 4),
        'PR-AUC': round(pr_auc, 4),
        'Fit Time (s)': round(t_elapsed, 2)
    })
    
    eval_dicts[name] = {
        'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc,
        'prec_curve': prec_curve, 'rec_curve': rec_curve, 'pr_auc': pr_auc,
        'cm': cm, 'y_proba': y_proba, 'y_pred': y_pred
    }

comparison_table = pd.DataFrame(results_list)
print("\\n" + "="*50)
print("ACTUAL VALIDATION BENCHMARK RESULTS")
print("="*50)
display(comparison_table)
"""))

cells.append(nbf.v4.new_code_cell("""# Diagnostic Curves: ROC Curve & PR Curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

colors = {'Logistic Regression': '#2563EB', 'Decision Tree': '#059669', 'Random Forest': '#D97706'}

# 1. ROC Curves
for name, data in eval_dicts.items():
    ax1.plot(data['fpr'], data['tpr'], label=f"{name} (AUC = {data['roc_auc']:.4f})", linewidth=2.2, color=colors[name])
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance (AUC = 0.5000)')
ax1.set_title("ROC Curves Comparison", fontsize=12, fontweight='bold')
ax1.set_xlabel("False Positive Rate", fontweight='bold')
ax1.set_ylabel("True Positive Rate (Recall)", fontweight='bold')
ax1.legend(loc='lower right')

# 2. Precision-Recall Curves
baseline = y_val.mean()
for name, data in eval_dicts.items():
    ax2.plot(data['rec_curve'], data['prec_curve'], label=f"{name} (PR-AUC = {data['pr_auc']:.4f})", linewidth=2.2, color=colors[name])
ax2.axhline(baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.3f})')
ax2.set_title("Precision-Recall Curves Comparison", fontsize=12, fontweight='bold')
ax2.set_xlabel("Recall", fontweight='bold')
ax2.set_ylabel("Precision", fontweight='bold')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# Confusion Matrices Side-by-Side
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

for ax, (name, data) in zip(axes, eval_dicts.items()):
    cm = data['cm']
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Safe (0)', 'Default (1)'], yticklabels=['Safe (0)', 'Default (1)'])
    ax.set_title(f"{name}\\nTN={cm[0,0]:,}, FP={cm[0,1]:,}\\nFN={cm[1,0]:,}, TP={cm[1,1]:,}", fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.suptitle("Normalized Confusion Matrices (Default Threshold = 0.50)", fontsize=13, fontweight='bold', y=1.03)
plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# SECTION 13: FEATURE IMPORTANCE
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## Section 13: Feature Importance Analysis
We analyze the Gini feature importances extracted from the champion **Random Forest** pipeline.  
Understanding which features drive predictions is vital in credit risk to ensure explainability and verify economic validity.
"""))

cells.append(nbf.v4.new_code_cell("""# Extract feature importances from Random Forest
rf_clf = rf_pipeline.named_steps['classifier']
engineer = rf_pipeline.named_steps['engineer']

# Generate feature names
dummy_sample = X_train.head(5)
engineered_sample = engineer.transform(dummy_sample)
feature_names = list(engineered_sample.columns)
importances = rf_clf.feature_importances_

df_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
df_importance = df_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_importance.head(10), x='Importance', y='Feature', palette='viridis')
plt.title("CreditGuard AI: Top 10 Predictive Features (Random Forest)", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Gini Importance Score", fontweight='bold')
plt.ylabel("Engineered Feature", fontweight='bold')
plt.show()

print("--- Top 5 Predictive Features ---")
for idx, row in df_importance.head(5).iterrows():
    print(f"- {row['Feature']}: {row['Importance']:.4f}")
"""))

# ==============================================================================
# SECTION 14 & 15: MODEL SAVING & PREDICTION FUNCTION
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## Section 14 & 15: Model Persistence & Interactive Prediction Function
We serialize the winning Random Forest pipeline using `joblib`.  
We then define an end-to-end prediction function that:
1. Accepts raw borrower features in standard currency and count units.
2. Applies the complete preprocessing and feature engineering pipeline.
3. Computes the probability of financial distress using `predict_proba()`.
4. Maps the probability into actionable risk tiers (Low, Moderate, High Risk).
"""))

cells.append(nbf.v4.new_code_cell("""# Save champion model to disk
model_path = MODELS_DIR / 'best_credit_model.joblib'
joblib.dump(rf_pipeline, model_path)
print(f"Persisted trained pipeline to: {model_path} ({model_path.stat().st_size / 1024 / 1024:.2f} MB)")

# Reusable prediction function
def predict_credit_risk(borrower_dict, pipeline, low_thresh=0.15, high_thresh=0.35):
    \"\"\"Predicts borrower financial distress probability and assigns risk category.\"\"\"
    df_borrower = pd.DataFrame([borrower_dict])
    prob = float(pipeline.predict_proba(df_borrower)[0, 1])
    
    if prob < low_thresh:
        category = "Low Risk (Tier 1)"
        assessment = "Strong financial stability. Low probability of default."
    elif prob <= high_thresh:
        category = "Moderate Risk (Tier 2)"
        assessment = "Moderate leverage or minor payment delays observed."
    else:
        category = "High Risk (Tier 3)"
        assessment = "Elevated risk profile. High credit stress or past severe delinquency."
        
    return {
        'Distress Probability': f"{prob:.2%}",
        'Raw Probability': round(prob, 4),
        'Risk Category': category,
        'Assessment': assessment
    }

# Test sample borrower profile
sample_borrower = {
    'RevolvingUtilizationOfUnsecuredLines': 0.28,
    'age': 38,
    'NumberOfTime30-59DaysPastDueNotWorse': 0,
    'DebtRatio': 0.22,
    'MonthlyIncome': 7500.0,
    'NumberOfOpenCreditLinesAndLoans': 9,
    'NumberOfTimes90DaysLate': 0,
    'NumberRealEstateLoansOrLines': 1,
    'NumberOfTime60-89DaysPastDueNotWorse': 0,
    'NumberOfDependents': 1
}

result = predict_credit_risk(sample_borrower, rf_pipeline)
print("--- Sample Prediction Output ---")
for k, v in result.items():
    print(f"{k}: {v}")
"""))

# ==============================================================================
# SECTION 16: KAGGLE TEST PREDICTIONS
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## Section 16: Kaggle Test Set Predictions
We load the unlabeled `cs-test.csv` dataset, pass it through the champion pipeline, and generate the probability predictions required for Kaggle competition submission (`Id,SeriousDlqin2yrs`).
"""))

cells.append(nbf.v4.new_code_cell("""test_file = DATA_DIR / 'cs-test.csv'
if test_file.exists():
    df_test_raw = pd.read_csv(test_file)
    print(f"Loaded test dataset: {len(df_test_raw):,} rows.")
    
    # Preserve original test IDs
    test_ids = df_test_raw['Unnamed: 0'] if 'Unnamed: 0' in df_test_raw.columns else np.arange(1, len(df_test_raw)+1)
    
    # Feature columns
    feature_cols = [c for c in df_test_raw.columns if c not in ['Unnamed: 0', 'SeriousDlqin2yrs']]
    X_test = df_test_raw[feature_cols]
    
    # Predict probabilities
    test_probs = rf_pipeline.predict_proba(X_test)[:, 1]
    
    # Create submission DataFrame
    sub_df = pd.DataFrame({'Id': test_ids, 'SeriousDlqin2yrs': test_probs})
    sub_path = OUTPUTS_DIR / 'credit_scoring_predictions.csv'
    sub_df.to_csv(sub_path, index=False)
    print(f"Saved Kaggle test predictions to: {sub_path}")
    print(f"Sample test predictions:\\n{sub_df.head()}")
else:
    print(f"cs-test.csv not found at {test_file}. Skipping test generation.")
"""))

# Assign cells to notebook and write out
nb['cells'] = cells
nb['metadata'] = {
    'language_info': {'name': 'python', 'version': '3.13'},
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
}

out_nb_path = PROJECT_ROOT / 'notebooks' / 'Credit_Scoring_Model.ipynb'
with open(out_nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Successfully generated complete Jupyter Notebook at: {out_nb_path}")
