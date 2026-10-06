"""
model_evaluator.py
Computes test metrics, classification reports, and confusion matrix visualisations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


def evaluate_models(trained_models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[pd.DataFrame, dict]:
    """
    Evaluates trained baseline models on the holdout test dataset.
    """
    print("\n[Model Evaluator] Evaluating Models on Test Set...")
    summary_list = []
    detailed_reports = {}

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        summary_list.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "F1_Macro": round(f1_macro, 4),
            "F1_Weighted": round(f1_weighted, 4)
        })
        
        detailed_reports[name] = {
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }
        
        print(f"  • {name:<22} | Test Acc: {acc:.4f} | Test F1-Macro: {f1_macro:.4f}")

    metrics_df = pd.DataFrame(summary_list).sort_values(by="F1_Macro", ascending=False).reset_index(drop=True)
    return metrics_df, detailed_reports


def plot_confusion_matrices(detailed_reports: dict, save_path: str = None) -> None:
    """
    Plots heatmaps for confusion matrices across all baseline models.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (name, report) in enumerate(detailed_reports.items()):
        cm = report["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Low', 'Medium', 'High'],
                    yticklabels=['Low', 'Medium', 'High'])
        axes[idx].set_title(f"Confusion Matrix: {name}")
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"[Model Evaluator] Saved confusion matrix figure to '{save_path}'.")
    plt.show()