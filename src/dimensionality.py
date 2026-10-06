import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.config_loader import NOMINAL_COLUMNS


def perform_pca_analysis(df_clean: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Performs scaling and PCA on continuous numerical features."""
    numeric_cols = [
        c for c in df_clean.columns 
        if c not in NOMINAL_COLUMNS + ['Family_Income_Level', 'Stress_Level']
    ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[numeric_cols])

    pca = PCA()
    pca.fit(X_scaled)

    cum_variance = np.cumsum(pca.explained_variance_ratio_)

    pca_df = pd.DataFrame({
        'Principal Component': [f'PC{i+1}' for i in range(len(numeric_cols))],
        'Explained Variance Ratio': pca.explained_variance_ratio_,
        'Cumulative Variance Ratio': cum_variance
    })

    print("\n" + "="*50)
    print("      PCA EXPLAINED VARIANCE RATIO")
    print("="*50)
    print(pca_df.to_string(index=False))

    # Plot PCA Cumulative Variance Scree Plot
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(numeric_cols) + 1), cum_variance, marker='o', linestyle='--', color='b')
    plt.axhline(y=0.90, color='r', linestyle=':', label='90% Variance Threshold')
    plt.title('PCA Cumulative Explained Variance', fontsize=13)
    plt.xlabel('Number of Principal Components', fontsize=11)
    plt.ylabel('Cumulative Variance Ratio', fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return pca_df