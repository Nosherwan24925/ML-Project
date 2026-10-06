import os
import pandas as pd
from src.config_loader import DATA_DIR, TARGET_COLUMN

# Define target file path
ENCODED_DATASET_NAME = "Encoded_DataSet.csv"
ENCODED_DATASET_PATH = os.path.join(DATA_DIR, ENCODED_DATASET_NAME)

SELECTED_DATASET_NAME = "Selected_Features_DataSet.csv"
SELECTED_DATASET_PATH = os.path.join(DATA_DIR, SELECTED_DATASET_NAME)


def save_encoded_dataset(df_encoded: pd.DataFrame, output_path: str = ENCODED_DATASET_PATH) -> str:
    """
    Saves the encoded DataFrame to CSV if it does not already exist.
    
    Parameters:
        df_encoded (pd.DataFrame): Preprocessed/Encoded dataset.
        output_path (str): Destination file path for the CSV.
        
    Returns:
        str: Path to the encoded dataset file.
    """
    if os.path.exists(output_path):
        print(f"[Exporter] Notice: '{ENCODED_DATASET_NAME}' already exists at '{output_path}'. Skipping file write.")
    else:
        # Ensure data directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save encoded DataFrame
        df_encoded.to_csv(output_path, index=False)
        print(f"[Exporter] Successfully exported encoded dataset ({df_encoded.shape[0]} rows, {df_encoded.shape[1]} columns) to: '{output_path}'")
        
    return output_path
def save_selected_dataset(df_encoded: pd.DataFrame, selected_features: list, output_path: str = SELECTED_DATASET_PATH) -> str:
    """
    Filters the encoded DataFrame to include only selected features + target column,
    and exports it to CSV if it does not already exist.
    """
    # Ensure target column is included in output
    output_cols = [col for col in selected_features if col in df_encoded.columns]
    if TARGET_COLUMN in df_encoded.columns and TARGET_COLUMN not in output_cols:
        output_cols.append(TARGET_COLUMN)

    df_selected = df_encoded[output_cols]

    if os.path.exists(output_path):
        print(f"[Exporter] Notice: '{SELECTED_DATASET_NAME}' already exists at '{output_path}'. Skipping file write.")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_selected.to_csv(output_path, index=False)
        print(f"[Exporter] Successfully exported selected dataset ({df_selected.shape[0]} rows, {df_selected.shape[1]} columns) to: '{output_path}'")

    return output_path