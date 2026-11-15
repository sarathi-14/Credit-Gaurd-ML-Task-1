"""Evaluation and reporting module for CreditGuard AI.

Calculates comprehensive classification metrics (ROC-AUC, PR-AUC, F1, Recall, Precision, Accuracy),
plots diagnostic curves, and writes evaluation artifacts.
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)
from src.utils import VIS_DIR, OUTPUTS_DIR


def evaluate_single_model(model, X_val, y_val, model_name="Model", threshold=0.5):
    """Compute detailed evaluation metrics for a single fitted pipeline.

    Parameters
    ----------
    model : Pipeline
        Fitted Scikit-learn pipeline with predict and predict_proba methods.
    X_val : pd.DataFrame
        Validation feature set.
    y_val : pd.Series or np.ndarray
        Ground truth binary labels.
    model_name : str
        Display name of the model.
    threshold : float
        Decision threshold for binary classification.

    Returns
    -------
    dict: dictionary containing all computed metrics, predictions, probabilities, and curves.
    """
    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_val, y_proba)
    pr_auc = average_precision_score(y_val, y_proba)
    cm = confusion_matrix(y_val, y_pred)

    fpr, tpr, _ = roc_curve(y_val, y_proba)
    precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_proba)

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "classification_report": classification_report(y_val, y_pred, digits=4),
        "y_proba": y_proba,
        "y_pred": y_pred,
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": precision_vals,
        "recall_curve": recall_vals,
    }


def plot_roc_curves(eval_results, save_path=None):
    """Plot ROC curves for multiple models on a unified chart."""
    plt.figure(figsize=(8, 6), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = ["#2563EB", "#059669", "#D97706", "#7C3AED"]

    for i, res in enumerate(eval_results):
        plt.plot(
            res["fpr"],
            res["tpr"],
            label=f"{res['model_name']} (ROC-AUC = {res['roc_auc']:.4f})",
            linewidth=2.2,
            color=colors[i % len(colors)],
        )

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    plt.title("CreditGuard AI: Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_pr_curves(eval_results, y_val, save_path=None):
    """Plot Precision-Recall curves for multiple models on a unified chart."""
    plt.figure(figsize=(8, 6), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = ["#2563EB", "#059669", "#D97706", "#7C3AED"]
    baseline = np.mean(y_val)

    for i, res in enumerate(eval_results):
        plt.plot(
            res["recall_curve"],
            res["precision_curve"],
            label=f"{res['model_name']} (PR-AUC = {res['pr_auc']:.4f})",
            linewidth=2.2,
            color=colors[i % len(colors)],
        )

    plt.axhline(y=baseline, color="k", linestyle="--", alpha=0.5, label=f"Class Imbalance Baseline ({baseline:.3f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=11, fontweight="bold")
    plt.ylabel("Precision", fontsize=11, fontweight="bold")
    plt.title("CreditGuard AI: Precision-Recall Curve Comparison", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices(eval_results, save_path=None):
    """Plot side-by-side normalized confusion matrices for all evaluated models."""
    n_models = len(eval_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), dpi=150)
    if n_models == 1:
        axes = [axes]

    for ax, res in zip(axes, eval_results):
        cm = res["confusion_matrix"]
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["No Delinquency (0)", "Delinquency (1)"],
            yticklabels=["No Delinquency (0)", "Delinquency (1)"],
            annot_kws={"size": 11, "fontweight": "bold"},
        )
        ax.set_title(f"{res['model_name']}\nRaw: TN={cm[0,0]}, FP={cm[0,1]}\nFN={cm[1,0]}, TP={cm[1,1]}", fontsize=10)
        ax.set_xlabel("Predicted Class", fontweight="bold")
        ax.set_ylabel("True Class", fontweight="bold")

    plt.suptitle("CreditGuard AI: Normalized Confusion Matrix Comparison", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_model_comparison_bar(comparison_df, save_path=None):
    """Plot grouped bar chart comparing key evaluation metrics across models."""
    metrics_to_plot = ["ROC-AUC", "PR-AUC", "F1-Score", "Recall", "Precision", "Accuracy"]
    df_melt = comparison_df.melt(id_vars=["Model"], value_vars=metrics_to_plot, var_name="Metric", value_name="Score")

    plt.figure(figsize=(10, 5.5), dpi=150)
    palette = sns.color_palette("muted", len(comparison_df["Model"].unique()))
    ax = sns.barplot(data=df_melt, x="Metric", y="Score", hue="Model", palette=palette)

    plt.title("CreditGuard AI: Model Performance Metrics Comparison", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Evaluation Metric", fontsize=11, fontweight="bold")
    plt.ylabel("Score", fontsize=11, fontweight="bold")
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right", frameon=True, fontsize=10)

    # Annotate values
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
                xytext=(0, 3),
                textcoords="offset points",
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_feature_importance(pipeline, top_n=10, save_path=None):
    """Plot horizontal feature importance for a tree-based pipeline model."""
    classifier = pipeline.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        return None

    feature_engineer = pipeline.named_steps["engineer"]
    feature_names = feature_engineer.get_feature_names_out()
    importances = classifier.feature_importances_

    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df_imp = df_imp.sort_values(by="Importance", ascending=False).head(top_n)

    plt.figure(figsize=(9, 5.5), dpi=150)
    colors = sns.color_palette("viridis", len(df_imp))
    plt.barh(df_imp["Feature"][::-1], df_imp["Importance"][::-1], color=colors, edgecolor="black", alpha=0.85)

    plt.title(f"CreditGuard AI: Top {top_n} Predictive Feature Importances", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Relative Gini Feature Importance", fontsize=11, fontweight="bold")
    plt.ylabel("Engineered Feature", fontsize=11, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    return df_imp


def save_evaluation_artifacts(eval_results, y_val, best_pipeline=None):
    """Compile all evaluation metrics, save comparison table, generate report, and save charts."""
    records = []
    for res in eval_results:
        records.append(
            {
                "Model": res["model_name"],
                "Accuracy": round(res["accuracy"], 4),
                "Precision": round(res["precision"], 4),
                "Recall": round(res["recall"], 4),
                "F1-Score": round(res["f1_score"], 4),
                "ROC-AUC": round(res["roc_auc"], 4),
                "PR-AUC": round(res["pr_auc"], 4),
            }
        )

    comparison_df = pd.DataFrame(records)
    csv_path = OUTPUTS_DIR / "model_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"Saved model comparison table to: {csv_path}")

    # Generate charts
    plot_roc_curves(eval_results, save_path=VIS_DIR / "roc_curves_comparison.png")
    plot_pr_curves(eval_results, y_val, save_path=VIS_DIR / "pr_curves_comparison.png")
    plot_confusion_matrices(eval_results, save_path=VIS_DIR / "confusion_matrices.png")
    plot_model_comparison_bar(comparison_df, save_path=VIS_DIR / "model_metrics_comparison.png")

    if best_pipeline:
        plot_feature_importance(best_pipeline, top_n=10, save_path=VIS_DIR / "feature_importance.png")

    # Generate comprehensive text report
    report_path = OUTPUTS_DIR / "evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("CREDITGUARD AI: MODEL EVALUATION & COMPARISON REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write("Task: Task 1 - Credit Scoring Model (Supervised Binary Classification)\n")
        f.write("Internship: CodeAlpha Machine Learning Internship\n")
        f.write("Target Variable: SeriousDlqin2yrs (0 = No Delinquency, 1 = Serious Delinquency)\n")
        f.write(f"Validation Records Evaluated: {len(y_val):,} (Class 0: {(y_val == 0).sum():,}, Class 1: {(y_val == 1).sum():,})\n\n")

        f.write("-" * 70 + "\n")
        f.write("1. SUMMARY COMPARISON TABLE\n")
        f.write("-" * 70 + "\n")
        f.write(comparison_df.to_string(index=False) + "\n\n")

        f.write("-" * 70 + "\n")
        f.write("2. DETAILED CLASSIFICATION REPORTS\n")
        f.write("-" * 70 + "\n")
        for res in eval_results:
            f.write(f"\n--- {res['model_name']} ---\n")
            f.write(f"ROC-AUC: {res['roc_auc']:.4f} | PR-AUC: {res['pr_auc']:.4f}\n")
            f.write(res["classification_report"] + "\n")
            cm = res["confusion_matrix"]
            f.write(f"Confusion Matrix:\n  TN: {cm[0,0]:>6}  |  FP: {cm[0,1]:>6}\n  FN: {cm[1,0]:>6}  |  TP: {cm[1,1]:>6}\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("3. BUSINESS & FINANCIAL RISK TRADEOFF ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write("In credit risk modeling, False Negatives (predicting a defaulting borrower as safe)\n")
        f.write("carry significantly higher financial costs than False Positives (flagging a safe borrower\n")
        f.write("for secondary review). Models trained with balanced class weights optimize the recall\n")
        f.write("and area under the ROC curve, ensuring high-risk profiles are effectively identified.\n\n")

    print(f"Saved comprehensive evaluation report to: {report_path}")
    return comparison_df
