"""
Catalyst Deactivation Kinetics & Long-Term Trajectory Simulator.
Author: Aditya001-max <kaushikadityaraj001@gmail.com>
GitHub: https://github.com/Aditya001-max
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple


def calculate_deactivation_rate(
    model: Any,
    sample_x_scaled: np.ndarray,
    baseline_time_h: float,
    target_time_h: float,
    initial_conversion: float,
    feature_names: list,
    x_scaler: Any,
    y_scaler: Any,
    transformed_cols: list
) -> Tuple[float, float]:
    """
    Simulates conversion at target_time_h and computes the deactivation rate:
    Rate = (Initial_Conv - Final_Conv) / (target_time_h - baseline_time_h)  [% / hour]
    """
    time_col_idx = feature_names.index('time_of_reaction_h')
    x_sim = sample_x_scaled.copy()

    # Re-apply log1p if time was transformed
    if 'time_of_reaction_h' in transformed_cols:
        x_sim[0, time_col_idx] = np.log1p(target_time_h)
    else:
        x_sim[0, time_col_idx] = target_time_h

    # Model inference
    if isinstance(model, nn.Module):
        model.eval()
        with torch.no_grad():
            t_in = torch.tensor(x_sim, dtype=torch.float32)
            pred_scaled = model(t_in).cpu().numpy()
    else:
        pred_scaled = model.predict(x_sim)

    pred_inv = y_scaler.inverse_transform(pred_scaled.reshape(1, -1))
    final_conv = pred_inv[0, 0]  # Conversion is at index 0

    dt = target_time_h - baseline_time_h
    rate = (initial_conversion - final_conv) / dt if dt != 0 else 0.0

    return float(final_conv), float(rate)


def simulate_deactivation_trajectories(
    models_dict: Dict[str, Any],
    sample_x_scaled: np.ndarray,
    feature_names: list,
    y_scaler: Any,
    transformed_cols: list,
    time_horizon: np.ndarray = np.linspace(0.5, 24.0, 50)
) -> Dict[str, Dict[str, list]]:
    """
    Computes dynamic conversion decay and coke accumulation curves over a continuous time horizon.
    """
    time_col_idx = feature_names.index('time_of_reaction_h')
    trajectories = {'conversion': {}, 'coke': {}}

    for m_name, model in models_dict.items():
        conv_list = []
        coke_list = []

        for t_val in time_horizon:
            x_sim = sample_x_scaled.copy()
            if 'time_of_reaction_h' in transformed_cols:
                x_sim[0, time_col_idx] = np.log1p(t_val)
            else:
                x_sim[0, time_col_idx] = t_val

            if isinstance(model, nn.Module):
                model.eval()
                with torch.no_grad():
                    t_in = torch.tensor(x_sim, dtype=torch.float32)
                    p_scaled = model(t_in).cpu().numpy()
            else:
                p_scaled = model.predict(x_sim)

            p_inv = y_scaler.inverse_transform(p_scaled.reshape(1, -1))
            conv_list.append(float(p_inv[0, 0]))
            coke_list.append(float(p_inv[0, 3]))

        trajectories['conversion'][m_name] = conv_list
        trajectories['coke'][m_name] = coke_list

    return trajectories
