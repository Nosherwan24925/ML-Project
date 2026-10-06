"""
export_results.py
Saves evaluation metrics and cross-validation summaries to disk.
"""

import os
import json
import pandas as pd
from src.config_loader import DATA_DIR

RESULTS_DIR = os.path.join(DATA_DIR, "results")


def save_baseline_results(cv_df: pd.DataFrame, test_metrics_df: pd.DataFrame, detailed_reports: dict) -> None:
    """
    Exports summary CSV files and JSON classification reports.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Save 10-Fold CV Metrics Summary
    cv_csv_path = os.path.join(RESULTS_DIR, "cross_validation_results.csv")
    cv_df.to_csv(cv_csv_path, index=False)
    print(f"[Exporter] Saved Stratified CV summary to '{cv_csv_path}'.")

    # Save Test Set Metrics Summary
    test_csv_path = os.path.join(RESULTS_DIR, "baseline_test_evaluation.csv")
    test_metrics_df.to_csv(test_csv_path, index=False)
    print(f"[Exporter] Saved test set evaluation metrics to '{test_csv_path}'.")

    # Save Detailed JSON Classification Reports & Confusion Matrices
    serializable_reports = {}
    for name, report in detailed_reports.items():
        serializable_reports[name] = {
            "classification_report": report["classification_report"],
            "confusion_matrix": report["confusion_matrix"].tolist()
        }

    reports_json_path = os.path.join(RESULTS_DIR, "detailed_model_reports.json")
    with open(reports_json_path, "w") as f:
        json.dump(serializable_reports, f, indent=4)
    print(f"[Exporter] Saved detailed JSON reports to '{reports_json_path}'.")