"""
Reaction Kinetics & Thermodynamic Principles for Propane Dehydrogenation (PDH).
Author: Aditya001-max <kaushikadityaraj001@gmail.com>
GitHub: https://github.com/Aditya001-max
"""

import numpy as np

# Universal Gas Constant (J/(mol*K))
R_GAS = 8.314

# Thermodynamic parameters for PDH: C3H8 <=> C3H6 + H2
DH_PDH = 124_000.0   # Enthalpy of reaction (J/mol, endothermic)
DS_PDH = 126.0       # Entropy of reaction (J/(mol*K))

# Activation energies (J/mol) based on literature kinetics (Li et al., Bai et al., Fogler)
EA_PDH = 90_000.0    # Activation energy for PDH
EA_COKE = 80_000.0   # Activation energy for coke deposition
EA_DEACT = 70_000.0  # Activation energy for catalyst deactivation

# Pre-exponential frequency factors
A_PDH = 1.0e7
A_COKE = 1.0e5

# Langmuir adsorption constants (bar^-1) and adsorption heats (J/mol)
K0_ADS = {'C3H8': 0.5, 'C3H6': 2.0, 'H2': 0.3}
DH_ADS = {'C3H8': -40000.0, 'C3H6': -60000.0, 'H2': -30000.0}


def arrhenius_rate(a_factor: float, e_act: float, temp_k: np.ndarray) -> np.ndarray:
    """Computes Arrhenius rate constant k = A * exp(-Ea / (R * T))."""
    t_clamped = np.clip(temp_k, 500.0, 1500.0)
    return a_factor * np.exp(-e_act / (R_GAS * t_clamped))


def langmuir_adsorption_const(species: str, temp_k: np.ndarray) -> np.ndarray:
    """Computes Langmuir adsorption equilibrium constant K_ads(T)."""
    return K0_ADS[species] * np.exp(-DH_ADS[species] / (R_GAS * temp_k))


def vant_hoff_equilibrium_const(temp_k: np.ndarray) -> np.ndarray:
    """Computes van't Hoff equilibrium constant K_eq(T) for C3H8 <=> C3H6 + H2."""
    return np.exp(-DH_PDH / (R_GAS * temp_k) + DS_PDH / R_GAS)


def equilibrium_conversion_limit(temp_k: np.ndarray, pressure_bar: float = 1.0) -> np.ndarray:
    """
    Computes thermodynamic equilibrium conversion ceiling X_eq(T) at pressure P:
    Kp = (X^2 * P) / (1 - X^2) => simplified for gas expansion.
    Returns conversion fraction in range [0.0, 0.999].
    """
    k_eq = vant_hoff_equilibrium_const(temp_k)
    # Solve quadratic: X_eq = (-K + sqrt(K^2 + 4K)) / 2 for P=1 bar
    x_eq = (-k_eq + np.sqrt(k_eq**2 + 4.0 * k_eq)) / 2.0
    return np.clip(x_eq, 0.0, 0.999)


def langmuir_hinshelwood_rate(temp_k: np.ndarray, p_c3h8: np.ndarray,
                              p_c3h6: float = 0.0, p_h2: float = 0.0) -> np.ndarray:
    """
    Langmuir-Hinshelwood rate [mol/(g_cat*s)] assuming 1st C-H bond cleavage is RDS.
    """
    k = arrhenius_rate(A_PDH, EA_PDH, temp_k)
    k_c8 = langmuir_adsorption_const('C3H8', temp_k)
    k_c6 = langmuir_adsorption_const('C3H6', temp_k)
    k_h2 = langmuir_adsorption_const('H2', temp_k)
    denom = (1.0 + k_c8 * p_c3h8 + k_c6 * p_c3h6 + k_h2 * p_h2)**2
    return k * k_c8 * p_c3h8 / denom


def fogler_deactivation_activity(time_h: np.ndarray, temp_k: np.ndarray,
                                 c_coke: np.ndarray, k_d0: float = 0.05,
                                 alpha: float = 0.8) -> np.ndarray:
    """
    Fogler power-law catalyst deactivation model:
    a(t) = exp(-kd(T) * (C_coke)^alpha * t)
    """
    k_d = arrhenius_rate(k_d0, EA_DEACT, temp_k)
    c_coke_clamped = np.clip(c_coke, 0.01, 20.0)
    return np.exp(-k_d * (c_coke_clamped**alpha) * time_h)
