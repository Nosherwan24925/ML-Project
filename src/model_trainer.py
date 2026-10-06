"""
model_trainer.py
Handles baseline classifier initialization, Stratified 10-Fold Cross-Validation,
and baseline training on scaled training splits.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def initialize_baseline_models(random_state: int = 42) -> dict:
    """
    Instantiates the suite of baseline classifiers for comparison.
    """
    return {
        "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "Support Vector Machine": SVC(random_state=random_state, probability=True)
    }


def perform_cross_validation(X: pd.DataFrame, y: pd.Series, models: dict, n_splits: int = 10, random_state: int = 42) -> pd.DataFrame:
    """
    Executes Stratified 10-Fold Cross-Validation using Scikit-Learn Pipelines.
    
    Why Pipeline? Ensures StandardScaler fits strictly on 9 training folds in each iteration 
    and transforms the 1 holdout fold, preventing data leakage across CV splits.
    """
    print(f"\n[Model Trainer] Executing Stratified {n_splits}-Fold Cross-Validation...")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = ['accuracy', 'f1_macro', 'f1_weighted']
    cv_summary = []

    for name, model in models.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        cv_scores = cross_validate(pipeline, X, y, cv=skf, scoring=scoring, return_train_score=False)
        
        mean_acc = cv_scores['test_accuracy'].mean()
        std_acc = cv_scores['test_accuracy'].std()
        mean_f1 = cv_scores['test_f1_macro'].mean()
        std_f1 = cv_scores['test_f1_macro'].std()
        mean_f1_w = cv_scores['test_f1_weighted'].mean()
        
        cv_summary.append({
            "Model": name,
            "CV_Mean_Accuracy": round(mean_acc, 4),
            "CV_Std_Accuracy": round(std_acc, 4),
            "CV_Mean_F1_Macro": round(mean_f1, 4),
            "CV_Std_F1_Macro": round(std_f1, 4),
            "CV_Mean_F1_Weighted": round(mean_f1_w, 4)
        })
        
        print(f"  • {name:<22} | CV Acc: {mean_acc:.4f} (±{std_acc:.4f}) | CV F1-Macro: {mean_f1:.4f} (±{std_f1:.4f})")

    cv_df = pd.DataFrame(cv_summary).sort_values(by="CV_Mean_F1_Macro", ascending=False).reset_index(drop=True)
    return cv_df


def train_models(models: dict, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Trains all baseline models on the pre-scaled training split (X_train_scaled, y_train).
    """
    trained_models = {}
    print("\n[Model Trainer] Training Baseline Models on Scaled Training Split...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"  ✓ {name} successfully trained.")
    
    return trained_models