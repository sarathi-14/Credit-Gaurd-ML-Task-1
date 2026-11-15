# Kaggle "Give Me Some Credit" Dataset

This directory contains the official dataset for the **Give Me Some Credit** competition hosted on Kaggle.

## Files Description

1. **`cs-training.csv`**:
   - Total rows: 150,000
   - Features: 11 financial and demographic attributes + 1 redundant index (`Unnamed: 0`)
   - Target: `SeriousDlqin2yrs` (Binary: 0 = No serious delinquency, 1 = Serious delinquency within 2 years)
   - Class distribution:
     - Class 0: 139,974 records (~93.32%)
     - Class 1: 10,026 records (~6.68%)

2. **`cs-test.csv`**:
   - Total rows: 101,503
   - Features: Identical to training set
   - Target: Unlabeled (used for scoring / probability predictions)

## Data Dictionary

| Variable Name | Description | Type | Value Range / Notes |
| :--- | :--- | :--- | :--- |
| **SeriousDlqin2yrs** | Person experienced 90 days past due delinquency or worse | Target (Binary Y/N) | 0, 1 |
| **RevolvingUtilizationOfUnsecuredLines** | Total balance on credit cards and personal lines of credit divided by the sum of credit limits | Percentage / Ratio | Skewed; values > 1 indicate over-limit |
| **age** | Age of borrower in years | Integer | Values range from 0 to 109 (0 is an invalid record to clean) |
| **NumberOfTime30-59DaysPastDueNotWorse** | Number of times borrower has been 30-59 days past due but no worse in the last 2 years | Integer | Contains extreme outlier codes (96, 98) |
| **DebtRatio** | Monthly debt payments, alimony, living costs divided by monthly gross income | Percentage / Ratio | When MonthlyIncome is missing, DebtRatio often holds absolute monthly debt |
| **MonthlyIncome** | Monthly income in currency units | Float | Missing for 29,731 rows (~19.82%) |
| **NumberOfOpenCreditLinesAndLoans** | Number of open loans (installment like car loan, mortgage) and lines of credit | Integer | Count of active credit accounts |
| **NumberOfTimes90DaysLate** | Number of times borrower has been 90 days or more past due | Integer | Severe delinquency indicator (contains 96, 98 codes) |
| **NumberRealEstateLoansOrLines** | Number of mortgage and real estate loans including home equity lines | Integer | Count of property-backed loans |
| **NumberOfTime60-89DaysPastDueNotWorse** | Number of times borrower has been 60-89 days past due but no worse in the last 2 years | Integer | Moderate delinquency indicator (contains 96, 98 codes) |
| **NumberOfDependents** | Number of dependents in family excluding borrower | Float/Integer | Missing for 3,924 rows (~2.62%) |
