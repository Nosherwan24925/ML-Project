import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif


def compute_shannon_entropy(series: pd.Series) -> float:
    """Calculates Shannon Entropy in bits."""
    if series.dtype != 'object' and len(series.unique()) > 10:
        counts = pd.cut(series, bins=10).value_counts()
    else:
        counts = series.value_counts()
    probs = counts / len(series)
    return float(entropy(probs, base=2))


def analyze_information_theory(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Calculates Entropy and Mutual Information scores for all features."""
    entropies = [compute_shannon_entropy(X[col]) for col in X.columns]
    mi_scores = mutual_info_classif(X, y, random_state=42)

    info_df = pd.DataFrame({
        'Feature': X.columns,
        'Entropy_Bits': entropies,
        'Mutual_Information': mi_scores
    }).sort_values(by='Mutual_Information', ascending=False)

    print("\n" + "="*50)
    print("      INFORMATION THEORY METRICS")
    print("="*50)
    print(info_df.to_string(index=False))

    # Plot Mutual Information
    plt.figure(figsize=(10, 6))
    sns.barplot(data=info_df, x='Mutual_Information', y='Feature', palette='mako')
    plt.title('Cross-Domain Feature Importance (Mutual Information)', fontsize=13)
    plt.xlabel('Mutual Information Score (Information Gain)', fontsize=11)
    plt.ylabel('Features', fontsize=11)
    plt.tight_layout()
    plt.show()

    return info_df