"""Model training and orchestration script for CreditGuard AI.

Trains Logistic Regression, Decision Tree, and Random Forest classifiers
inside end-to-end Scikit-learn pipelines, evaluates performance,
and persists the champion model.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.utils import load_dataset, MODELS_DIR
from src.preprocessing import (
    split_features_and_target,
    get_stratified_train_val_split,
    create_model_pipeline,
)
from src.evaluate_model import (
    evaluate_single_model,
    save_evaluation_artifacts,
)


def train_and_evaluate_all():
    """Execute complete model training and benchmarking pipeline."""
    print("=" * 70)
    print("CREDITGUARD AI: MODEL TRAINING & BENCHMARKING PIPELINE")
    print("=" * 70)

    # 1. Load Data
    print("\n[Step 1/6] Loading cs-training.csv...")
    df = load_dataset("cs-training.csv")
    print(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns.")

    # 2. Split Features & Target
    print("\n[Step 2/6] Separating features and target...")
    X, y = split_features_and_target(df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution:\n  Class 0 (No Delinquency): {(y == 0).sum():,} ({((y == 0).mean()*100):.2f}%)\n  Class 1 (Serious Delinquency): {(y == 1).sum():,} ({((y == 1).mean()*100):.2f}%)")

    # 3. Stratified Split
    print("\n[Step 3/6] Performing Stratified Train-Validation Split (80/20)...")
    X_train, X_val, y_train, y_val = get_stratified_train_val_split(X, y, test_size=0.20, random_state=42)
    print(f"Training set: {X_train.shape[0]:,} records")
    print(f"Validation set: {X_val.shape[0]:,} records")

    # 4. Define Candidate Model Pipelines
    print("\n[Step 4/6] Initializing Model Pipelines with Balanced Class Weights...")
    models = {
        "Logistic Regression": {
            "pipeline": create_model_pipeline(
                classifier=LogisticRegression(
                    C=1.0,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
                use_scaler=True,  # Scaling is required for gradient/L-BFGS optimization
            ),
            "description": "Baseline linear classifier with L2 regularization and RobustScaler.",
        },
        "Decision Tree": {
            "pipeline": create_model_pipeline(
                classifier=DecisionTreeClassifier(
                    max_depth=6,
                    min_samples_split=25,
                    min_samples_leaf=15,
                    class_weight="balanced",
                    random_state=42,
                ),
                use_scaler=False,  # Trees are invariant to monotonic feature scaling
            ),
            "description": "Pruned single decision tree with controlled depth to prevent overfitting.",
        },
        "Random Forest": {
            "pipeline": create_model_pipeline(
                classifier=RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=42,
                ),
                use_scaler=False,  # Ensemble of decision trees
            ),
            "description": "Bagged ensemble of 100 trees with controlled depth and bootstrap aggregation.",
        },
    }

    # 5. Fit & Evaluate Models
    print("\n[Step 5/6] Training & Evaluating Candidate Models...")
    eval_results = []
    trained_pipelines = {}

    for model_name, config in models.items():
        print(f"\n---> Training: {model_name}...")
        start_time = time.time()
        pipeline = config["pipeline"]
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - start_time
        print(f"     Fit completed in {elapsed:.2f} seconds.")

        print(f"     Evaluating on held-out validation set ({len(y_val):,} records)...")
        eval_dict = evaluate_single_model(pipeline, X_val, y_val, model_name=model_name, threshold=0.5)
        eval_dict["training_time_seconds"] = round(elapsed, 2)
        eval_results.append(eval_dict)
        trained_pipelines[model_name] = pipeline

        print(
            f"     Results -> Accuracy: {eval_dict['accuracy']:.4f} | "
            f"Precision: {eval_dict['precision']:.4f} | "
            f"Recall: {eval_dict['recall']:.4f} | "
            f"F1-Score: {eval_dict['f1_score']:.4f} | "
            f"ROC-AUC: {eval_dict['roc_auc']:.4f} | "
            f"PR-AUC: {eval_dict['pr_auc']:.4f}"
        )

    # 6. Select Champion Model based on ROC-AUC
    print("\n[Step 6/6] Selecting Champion Model & Saving Artifacts...")
    best_result = max(eval_results, key=lambda x: x["roc_auc"])
    best_model_name = best_result["model_name"]
    best_pipeline = trained_pipelines[best_model_name]
    print(f"\n*** Champion Model: {best_model_name} (ROC-AUC = {best_result['roc_auc']:.4f}) ***")

    # Persist Best Pipeline
    model_save_path = MODELS_DIR / "best_credit_model.joblib"
    joblib.dump(best_pipeline, model_save_path)
    print(f"Saved best model pipeline to: {model_save_path}")

    # Save Evaluation Artifacts (Charts, CSV comparison, text report)
    comparison_df = save_evaluation_artifacts(eval_results, y_val, best_pipeline=best_pipeline)

    # Save Model Metadata JSON for Streamlit Frontend integration
    metadata = {
        "best_model_name": best_model_name,
        "validation_roc_auc": round(best_result["roc_auc"], 4),
        "validation_pr_auc": round(best_result["pr_auc"], 4),
        "validation_f1_score": round(best_result["f1_score"], 4),
        "validation_recall": round(best_result["recall"], 4),
        "validation_precision": round(best_result["precision"], 4),
        "validation_accuracy": round(best_result["accuracy"], 4),
        "training_records": len(X_train),
        "validation_records": len(X_val),
        "model_file": str(model_save_path.name),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics_summary": comparison_df.to_dict(orient="records"),
    }
    meta_path = MODELS_DIR / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved model metadata to: {meta_path}")

    print("\n" + "=" * 70)
    print("ALL TRAINING & EVALUATION STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    return best_pipeline, comparison_df, eval_results


if __name__ == "__main__":
    train_and_evaluate_all()
