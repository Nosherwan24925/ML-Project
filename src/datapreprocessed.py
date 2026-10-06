import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config_loader import TARGET_COLUMN, DATA_DIR

SELECTED_DATASET_NAME = "Selected_Features_DataSet.csv"
SELECTED_DATASET_PATH = os.path.join(DATA_DIR, SELECTED_DATASET_NAME)

PROCESSED_DATASET_NAME = "DataProcessed.csv"
PROCESSED_DATASET_PATH = os.path.join(DATA_DIR, PROCESSED_DATASET_NAME)


def load_selected_dataset(file_path: str = SELECTED_DATASET_PATH) -> pd.DataFrame:
    """
    0. Loads the Selected_Features_DataSet.csv file from the data directory.
    Checks if the file exists before attempting to read.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"[Loader Error] Selected dataset file not found at '{file_path}'. "
            "Please ensure Phase 2 - Objective 1 feature selection step has run."
        )
    
    df = pd.read_csv(file_path)
    print(f"[Loader] Successfully loaded '{os.path.basename(file_path)}' with shape {df.shape}.")
    return df


def remove_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """1. Removes rows containing any missing (NaN/null) values."""
    initial_rows = df.shape[0]
    df_clean = df.dropna().reset_index(drop=True)
    dropped_count = initial_rows - df_clean.shape[0]
    print(f"[Preprocessing] Missing values check complete. Dropped {dropped_count} rows containing NaNs.")
    return df_clean


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """2. Removes duplicate rows from the dataset."""
    initial_rows = df.shape[0]
    df_clean = df.drop_duplicates().reset_index(drop=True)
    dropped_count = initial_rows - df_clean.shape[0]
    print(f"[Preprocessing] Duplicate check complete. Dropped {dropped_count} duplicate rows.")
    return df_clean


def export_processed_dataset(df_clean: pd.DataFrame, output_path: str = PROCESSED_DATASET_PATH) -> str:
    """
    3. Saves the cleaned preprocessed DataFrame to DataProcessed.csv if it does not already exist.
    """
    if os.path.exists(output_path):
        print(f"[Exporter] Notice: '{PROCESSED_DATASET_NAME}' already exists at '{output_path}'. Skipping file write.")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_clean.to_csv(output_path, index=False)
        print(f"[Exporter] Successfully exported cleaned dataset ({df_clean.shape[0]} rows, {df_clean.shape[1]} columns) to: '{output_path}'")
    return output_path


def perform_stratified_split(
    df: pd.DataFrame, 
    target_col: str = TARGET_COLUMN, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    4. Performs Stratified Train-Test Split.
    
    Why Stratified? Target class 2 (High Stress) represents ~10.9% of the dataset.
    Stratification guarantees that both training and testing sets maintain the 
    exact class distribution (Low/Medium/High), preventing class imbalance distortion.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )

    print(f"[Preprocessing] Stratified split complete (Train: {len(X_train)}, Test: {len(X_test)}).")
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    5. Applies Standard Scaling (Z-score normalization).
    
    Why Necessary? Features have different ranges (e.g., Screen_Time 1-11 vs Sleep_Hours 4-9).
    Scales features to zero mean and unit variance using parameters fitted ONLY on X_train 
    to prevent data leakage into X_test.
    """
    scaler = StandardScaler()
    
    # Fit scaler on training data and transform train set
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=X_train.columns, 
        index=X_train.index
    )
    
    # Transform test set using fitted training parameters
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=X_test.columns, 
        index=X_test.index
    )

    print("[Preprocessing] Standard scaling applied successfully (fit on train, transformed test).")
    return X_train_scaled, X_test_scaled, scaler


def run_preprocessing_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    Master pipeline function executing loading, cleaning, export, splitting, and scaling sequentially.
    """
    print("\n" + "="*50)
    print("      DATA PREPROCESSING PIPELINE (PHASE 2 - OBJ 2)      ")
    print("="*50)

    # Step 0: Load Selected Features Dataset
    df_selected = load_selected_dataset()

    # Step 1: Remove missing rows
    df_no_missing = remove_missing_rows(df_selected)

    # Step 2: Remove duplicate rows
    df_clean = remove_duplicate_rows(df_no_missing)

    # Step 3: Export cleaned dataset to data/DataProcessed.csv (Checks if file exists)
    export_processed_dataset(df_clean)

    # Step 4: Perform Stratified Train-Test Split
    X_train, X_test, y_train, y_test = perform_stratified_split(df_clean)

    # Step 5: Scale Features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print("="*50 + "\n")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler