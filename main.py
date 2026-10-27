"""
Main Execution Driver: Catalyst Screening & Deactivation Modeling Pipeline
Author: Aditya001-max <kaushikadityaraj001@gmail.com>
GitHub: https://github.com/Aditya001-max
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.preprocessing import preprocess_pdh_dataset, TARGET_COLS
from src.kinetics import equilibrium_conversion_limit
from src.models import (
    train_physics_informed_ann,
    build_random_forest,
    build_xgboost,
    build_catboost
)
from src.evaluate import compute_metrics, generate_benchmark_matrix
from src.deactivation import calculate_deactivation_rate, simulate_deactivation_trajectories


def load_dataset(data_path: str = None) -> pd.DataFrame:
    """Loads PDH experimental or benchmark dataset."""
    if data_path and os.path.exists(data_path):
        print(f"[DataLoader] Loading specified dataset: {data_path}")
        if data_path.endswith('.xlsx'):
            return pd.read_excel(data_path)
        return pd.read_csv(data_path)

    # Check for default benchmark csv
    default_csv = os.path.join(os.path.dirname(__file__), "data", "pdh_catalyst_data_sample.csv")
    if os.path.exists(default_csv):
        print(f"[DataLoader] Loading benchmark dataset: {default_csv}")
        return pd.read_csv(default_csv)

    # Generate synthetic if not found
    print("[DataLoader] Generating 713-sample benchmark dataset...")
    from data.generate_sample_data import generate_benchmark_pdh_data
    df_gen = generate_benchmark_pdh_data(713)
    df_gen.to_csv(default_csv, index=False)
    return df_gen


def run_pipeline(data_path: str = None, epochs: int = 50, batch_size: int = 32):
    print("=" * 70)
    print("  CATALYST SCREENING & PHYSICS-INFORMED ML PIPELINE")
    print("  Author: Aditya001-max | GitHub: https://github.com/Aditya001-max")
    print("=" * 70)

    # 1. Data Ingestion
    df = load_dataset(data_path)
    print(f"Dataset shape: {df.shape} ({len(df)} samples, {df.shape[1]} columns)")

    # 2. Preprocessing & Skewness Mitigation
    print("\n[Step 1/4] Preprocessing & Skewness Rectification...")
    prep = preprocess_pdh_dataset(df, skew_threshold=0.75, test_size=0.2)
    x_train, x_test = prep['X_train'], prep['X_test']
    y_train, y_test = prep['y_train'], prep['y_test']
    x_scaler, y_scaler = prep['x_scaler'], prep['y_scaler']
    feat_names = prep['feature_names']
    target_names = prep['target_names']
    transformed_cols = prep['transformed_skew_cols']

    print(f"Transformed right-skewed features via log1p: {transformed_cols}")
    print(f"Features dimension post-encoding: {x_train.shape[1]}")

    # 3. Compute Thermodynamic Limits for Physics Loss
    # Split train into train/val
    x_ann_tr, x_ann_vl, y_ann_tr, y_ann_vl, idx_ann_tr, idx_ann_vl = train_test_split(
        x_train, y_train, prep['idx_train'], test_size=0.2, random_state=42
    )

    t_tr_k = df.loc[idx_ann_tr, 'temperature_C'].values + 273.15
    xeq_raw_tr = equilibrium_conversion_limit(t_tr_k) * 100.0
    conv_min, conv_max = y_scaler.data_min_[0], y_scaler.data_max_[0]
    xeq_scaled_tr = (xeq_raw_tr - conv_min) / (conv_max - conv_min + 1e-8)

    t_vl_k = df.loc[idx_ann_vl, 'temperature_C'].values + 273.15
    xeq_raw_vl = equilibrium_conversion_limit(t_vl_k) * 100.0
    xeq_scaled_vl = (xeq_raw_vl - conv_min) / (conv_max - conv_min + 1e-8)

    # 4. Model Training
    print("\n[Step 2/4] Training Multi-Model Architecture Suite...")

    # A. Random Forest
    print("  -> Training Random Forest Regressor...")
    rf_model = build_random_forest()
    rf_model.fit(x_train, y_train)

    # B. XGBoost
    print("  -> Training XGBoost Regressor...")
    xgb_model = build_xgboost()
    xgb_model.fit(x_train, y_train)

    # C. CatBoost
    print("  -> Training CatBoost Regressor (MultiRMSE)...")
    cb_model = build_catboost()
    cb_model.fit(x_train, y_train)

    # D. Physics-Informed ANN
    print(f"  -> Training Physics-Informed Neural Network (PI-ANN, {epochs} epochs)...")
    pi_ann, tr_hist, val_hist = train_physics_informed_ann(
        x_ann_tr, y_ann_tr, xeq_scaled_tr,
        x_ann_vl, y_ann_vl, xeq_scaled_vl,
        epochs=epochs, batch_size=batch_size
    )

    # 5. Model Evaluation
    print("\n[Step 3/4] Evaluating Predictions & Compiling Benchmarks...")
    y_test_inv = y_scaler.inverse_transform(y_test)

    # Predictions
    rf_pred = y_scaler.inverse_transform(rf_model.predict(x_test))
    xgb_pred = y_scaler.inverse_transform(xgb_model.predict(x_test))
    cb_pred = y_scaler.inverse_transform(cb_model.predict(x_test))

    pi_ann.eval()
    with torch.no_grad():
        pi_pred_scaled = pi_ann(torch.tensor(x_test, dtype=torch.float32)).numpy()
    pi_pred = y_scaler.inverse_transform(pi_pred_scaled)

    # Enforce thermodynamic ceiling on test predictions
    t_test_k = df.loc[prep['idx_test'], 'temperature_C'].values + 273.15
    xeq_test = equilibrium_conversion_limit(t_test_k) * 100.0
    pi_pred[:, 0] = np.clip(pi_pred[:, 0], 0.0, xeq_test + 5.0)

    models_preds = {
        'Random Forest': rf_pred,
        'XGBoost': xgb_pred,
        'CatBoost': cb_pred,
        'PI-ANN (Ours)': pi_pred
    }

    benchmark_matrix = generate_benchmark_matrix(models_preds, y_test_inv, target_names)
    print("\n--- MASTER COMPARATIVE PERFORMANCE MATRIX ---")
    print(benchmark_matrix.to_string(index=False))

    # 6. Deactivation Simulation
    print("\n[Step 4/4] Simulating Catalyst Deactivation Kinetics (t = 0 to 24h)...")
    sample_idx = 0
    sample_scaled = x_test[sample_idx:sample_idx+1]
    orig_idx = prep['idx_test'][sample_idx]
    init_conv = df.loc[orig_idx, 'Conversion_%']
    base_time = df.loc[orig_idx, 'time_of_reaction_h']

    deact_results = []
    models_dict = {'Random Forest': rf_model, 'XGBoost': xgb_model, 'CatBoost': cb_model, 'PI-ANN': pi_ann}

    for m_name, mod in models_dict.items():
        f_conv, rate = calculate_deactivation_rate(
            mod, sample_scaled, base_time, 10.0, init_conv, feat_names, x_scaler, y_scaler, transformed_cols
        )
        deact_results.append({
            'Model': m_name,
            'Initial Time (h)': round(base_time, 1),
            'Baseline Conv (%)': round(init_conv, 2),
            'Conv @ 10h (%)': round(f_conv, 2),
            'Deactivation Rate (%/h)': round(rate, 4)
        })

    deact_df = pd.DataFrame(deact_results)
    print("\n--- SIMULATED DEACTIVATION KINETICS SUMMARY ---")
    print(deact_df.to_string(index=False))
    print("\nPipeline execution complete successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Catalyst Screening & Deactivation Modeling Pipeline")
    parser.add_argument("--data", type=str, default=None, help="Path to input Excel or CSV dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs for PI-ANN")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for PI-ANN")
    args = parser.parse_args()

    run_pipeline(args.data, args.epochs, args.batch)
