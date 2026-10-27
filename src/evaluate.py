"""
Model Evaluation, Benchmark Comparison, and Validation Metrics.
Author: Aditya001-max <kaushikadityaraj001@gmail.com>
GitHub: https://github.com/Aditya001-max
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Dict, Any


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: list) -> pd.DataFrame:
    """Computes R2, MAE, RMSE, and MSE for multi-output regression predictions."""
    records = []
    for i, name in enumerate(target_names):
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mse)
        records.append({
            'Target': name,
            'R2': round(float(r2), 4),
            'MAE': round(float(mae), 4),
            'RMSE': round(float(rmse), 4),
            'MSE': round(float(mse), 4)
        })
    return pd.DataFrame(records)


def generate_benchmark_matrix(models_preds: Dict[str, np.ndarray], y_true: np.ndarray, target_names: list) -> pd.DataFrame:
    """
    Creates a master comparative benchmark across all evaluated architectures.
    """
    all_rows = []
    for target_idx, target_name in enumerate(target_names):
        for metric in ['R2', 'MAE', 'RMSE']:
            row = {'Target': target_name, 'Metric': metric}
            for model_name, y_pred in models_preds.items():
                if metric == 'R2':
                    val = r2_score(y_true[:, target_idx], y_pred[:, target_idx])
                elif metric == 'MAE':
                    val = mean_absolute_error(y_true[:, target_idx], y_pred[:, target_idx])
                elif metric == 'RMSE':
                    val = np.sqrt(mean_squared_error(y_true[:, target_idx], y_pred[:, target_idx]))
                row[model_name] = round(float(val), 4)
            all_rows.append(row)
    return pd.DataFrame(all_rows)
