"""
Data Preprocessing, Skewness Mitigation, and Feature Engineering Pipeline.
Author: Aditya001-max <kaushikadityaraj001@gmail.com>
GitHub: https://github.com/Aditya001-max
"""

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Any

TARGET_COLS = ['Conversion_%', 'Selectivity_%', 'Propene_yield%', 'Coke_deposition_wt%']


def analyze_skewness(df: pd.DataFrame, threshold: float = 0.75) -> Tuple[pd.Series, List[str]]:
    """Identifies numerical columns exceeding the skewness threshold."""
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    skew_series = df[num_cols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    high_skew_cols = skew_series[skew_series > threshold].index.tolist()
    return skew_series, high_skew_cols


def preprocess_pdh_dataset(
    df: pd.DataFrame,
    skew_threshold: float = 0.75,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Full preprocessing pipeline:
    1. log1p transformation for highly skewed features
    2. One-hot encoding for categorical variables (Catalyst, Dopants)
    3. Train-test splitting
    4. Min-Max normalization for neural network stability
    """
    df_proc = df.copy()

    # 1. Skewness correction
    _, high_skew_cols = analyze_skewness(df_proc, threshold=skew_threshold)
    for col in high_skew_cols:
        df_proc[col] = np.log1p(df_proc[col])

    # 2. Categorical One-Hot Encoding
    cat_cols = df_proc.select_dtypes(include='object').columns.tolist()
    if cat_cols:
        df_proc = pd.get_dummies(df_proc, columns=cat_cols, drop_first=True)

    # 3. Separation of inputs and multi-target vector
    targets = [col for col in TARGET_COLS if col in df_proc.columns]
    x_features = df_proc.drop(columns=targets)
    y_targets = df_proc[targets]

    # 4. Scalers
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x_scaled = x_scaler.fit_transform(x_features)
    y_scaled = y_scaler.fit_transform(y_targets)

    # 5. Train/Test Split
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        x_scaled, y_scaled, df.index, test_size=test_size, random_state=random_state
    )

    return {
        'df_processed': df_proc,
        'X_train': x_train,
        'X_test': x_test,
        'y_train': y_train,
        'y_test': y_test,
        'idx_train': idx_train,
        'idx_test': idx_test,
        'x_scaler': x_scaler,
        'y_scaler': y_scaler,
        'feature_names': x_features.columns.tolist(),
        'target_names': targets,
        'transformed_skew_cols': high_skew_cols
    }
