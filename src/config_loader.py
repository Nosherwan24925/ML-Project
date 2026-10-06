import os
import pandas as pd

# Directory Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_NAME = "DataSet_StressLevel_Prediction.csv"
DATASET_PATH = os.path.join(DATA_DIR, DATASET_NAME)

# Target & Leakage Columns
TARGET_COLUMN = "Stress_Level"
LEAKAGE_COLUMNS = ["Timestamp", "Stress_Score"]

# Mapping Dictionaries
ORDINAL_MAP = {'Low': 0, 'Medium': 1, 'High': 2}

# Categorical Nominal Columns
NOMINAL_COLUMNS = [
    'Gender', 
    'Tuition (Private Coaching)', 
    'Physical_Exercise (hours per week)', 
    'University_Type'
]

# Domain Pillars
DOMAINS = {
    'Demographics': ['Age', 'Gender', 'University_Type'],
    'Academic': [
        'Study_Hours (per day)', 'Class_Attendance (%)', 
        'Tuition (Private Coaching)', 'Exam_Frequency (per semester)', 
        'Assignment_Load'
    ],
    'Lifestyle': [
        'Sleep_Hours (per day)', 'Physical_Exercise (hours per week)', 
        'Social_Media_Use (hours per day)', 'Screen_Time (hours per day)'
    ],
    'Socioeconomic_Psychological': [
        'Family_Income_Level', 'Peer_Pressure', 
        'Family_Support', 'Anxiety_Level'
    ]
}


def load_raw_dataset(file_path: str = DATASET_PATH) -> pd.DataFrame:
    """Loads CSV dataset and cleans column headers and leakage columns."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    df = pd.read_csv(file_path)

    # Clean header formatting (strip numbering prefixes and newlines)
    df.columns = [
        c.split('.')[1].split('\n')[0].strip() if '.' in c else c.strip() 
        for c in df.columns
    ]

    # Remove target leakage and metadata
    cols_to_drop = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    df_clean = df.drop(columns=cols_to_drop)

    print(f"[DataLoader] Loaded dataset from {file_path}. Cleaned Shape: {df_clean.shape}")
    return df_clean