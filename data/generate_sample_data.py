"""
Benchmark Synthetic PDH Dataset Generator (713 Samples).
Author: Aditya001-max <kaushikadityaraj001@gmail.com>
GitHub: https://github.com/Aditya001-max
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.kinetics import equilibrium_conversion_limit, arrhenius_rate, A_PDH, EA_PDH, A_COKE, EA_COKE


def generate_benchmark_pdh_data(n_samples: int = 713, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a physically consistent experimental screening dataset for Propane Dehydrogenation.
    Target values conform to thermodynamic equilibrium bounds and Langmuir kinetics.
    """
    np.random.seed(random_seed)

    # Process conditions
    time_h = np.random.exponential(scale=3.5, size=n_samples) + 0.5  # Skewed reaction time
    temp_c = np.random.uniform(520.0, 630.0, size=n_samples)
    temp_k = temp_c + 273.15
    propane_h2_ratio = np.random.uniform(0.5, 4.0, size=n_samples)
    p_propane = np.random.uniform(0.1, 0.9, size=n_samples)

    # Catalyst morphology & composition
    bet_surface = np.random.lognormal(mean=4.8, sigma=0.5, size=n_samples)  # Skewed BET area
    pt_loading = np.random.lognormal(mean=-0.5, sigma=0.4, size=n_samples).clip(0.1, 2.5)
    pore_size = np.random.uniform(2.5, 14.0, size=n_samples)
    pore_vol = (bet_surface * pore_size * 1e-4) * np.random.uniform(0.8, 1.2, size=n_samples)
    reduction_temp = np.random.uniform(380.0, 520.0, size=n_samples)

    catalysts = ['Pt/Al2O3', 'Pt-Sn/Al2O3', 'Pt-In/Al2O3', 'Pt-Ga/Al2O3', 'Pt/SiO2', 'Pt/ZSM-5']
    dopants = ['None', 'Sn', 'In', 'Ga', 'Ce', 'K']

    cat_choice = np.random.choice(catalysts, size=n_samples)
    dop_choice = np.random.choice(dopants, size=n_samples)

    # Thermodynamic ceiling
    x_eq = equilibrium_conversion_limit(temp_k, pressure_bar=1.0) * 100.0

    # Kinetic reaction calculation
    k_rxn = arrhenius_rate(A_PDH, EA_PDH, temp_k)
    k_coke = arrhenius_rate(A_COKE, EA_COKE, temp_k)

    # Conversion (%)
    cat_activity = (pt_loading * 0.4 + bet_surface * 0.002)
    conversion = (x_eq * (1.0 - np.exp(-k_rxn * cat_activity * (1.0 / (1.0 + time_h * 0.15))))).clip(5.0, 95.0)
    # Add experimental noise
    conversion = (conversion + np.random.normal(0, 1.5, size=n_samples)).clip(5.0, x_eq)

    # Selectivity (%)
    base_selectivity = 96.0 - (temp_c - 520.0) * 0.08 - (1.0 / (propane_h2_ratio + 0.1)) * 3.0
    selectivity = (base_selectivity + np.random.normal(0, 2.0, size=n_samples)).clip(45.0, 98.0)

    # Propene Yield (%)
    propene_yield = (conversion * selectivity / 100.0)

    # Coke deposition (wt%) - cumulative with time, accelerated by temperature and suppressed by H2
    coke_rate = k_coke * (p_propane / (1.0 + propane_h2_ratio * 0.5)) * (bet_surface * 0.005)
    coke = (coke_rate * np.sqrt(time_h) * 10.0 + np.random.normal(0, 0.2, size=n_samples)).clip(0.1, 15.0)

    df = pd.DataFrame({
        'time_of_reaction_h': time_h,
        'temperature_C': temp_c,
        'Propane_H2_ratio': propane_h2_ratio,
        'Partial_pressure_propane_bar': p_propane,
        'Surface_BET_m2_g': bet_surface,
        'Pt_loading_wt%': pt_loading,
        'Pore_size_nm': pore_size,
        'Pore_volume_cm3_g': pore_vol,
        'Reduction_Temperature_TPR_C': reduction_temp,
        'Catalyst': cat_choice,
        'Dopants_Pt-M': dop_choice,
        'Conversion_%': conversion,
        'Selectivity_%': selectivity,
        'Propene_yield%': propene_yield,
        'Coke_deposition_wt%': coke
    })

    return df


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_file = os.path.join(out_dir, "pdh_catalyst_data_sample.csv")
    df_sample = generate_benchmark_pdh_data(713)
    df_sample.to_csv(out_file, index=False)
    print(f"Generated {len(df_sample)} benchmark PDH records saved to {out_file}")
