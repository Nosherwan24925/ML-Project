import pandas as pd
from src.config_loader import DOMAINS, ORDINAL_MAP, NOMINAL_COLUMNS, TARGET_COLUMN


def summarize_domains(df: pd.DataFrame) -> None:
    """Prints a structured summary of features categorized by domain."""
    print("\n" + "="*50)
    print("      DOMAIN CHARACTERIZATION SUMMARY")
    print("="*50)
    for domain_name, features in DOMAINS.items():
        print(f"\n[{domain_name.upper()} DOMAIN] - ({len(features)} Features):")
        for f in features:
            if f in df.columns:
                print(f"  • {f:<38} | Type: {df[f].dtype}")


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Applies ordinal mapping, one-hot encoding, and splits features (X) and target (y)."""
    df_encoded = df.copy()

    # Apply Ordinal Transformations
    if 'Family_Income_Level' in df_encoded.columns:
        df_encoded['Family_Income_Level'] = df_encoded['Family_Income_Level'].map(ORDINAL_MAP)

    if TARGET_COLUMN in df_encoded.columns:
        df_encoded[TARGET_COLUMN] = df_encoded[TARGET_COLUMN].map(ORDINAL_MAP)

    # Apply One-Hot Encoding to Nominal Variables
    existing_nominal = [c for c in NOMINAL_COLUMNS if c in df_encoded.columns]
    df_encoded = pd.get_dummies(df_encoded, columns=existing_nominal, drop_first=True)

    # Separate Predictor Matrix (X) and Target Vector (y)
    X = df_encoded.drop(columns=[TARGET_COLUMN])
    y = df_encoded[TARGET_COLUMN]

    print(f"[Encoder] Categorical encoding complete. Predictor Matrix (X) Shape: {X.shape}")
    return X, y, df_encoded