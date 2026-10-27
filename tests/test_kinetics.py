"""
Unit Tests for Reaction Kinetics & Thermodynamic Boundary Constraints.
Author: Aditya001-max <kaushikadityaraj001@gmail.com>
GitHub: https://github.com/Aditya001-max
"""

import numpy as np
import pytest
from src.kinetics import (
    arrhenius_rate,
    vant_hoff_equilibrium_const,
    equilibrium_conversion_limit,
    langmuir_adsorption_const,
    langmuir_hinshelwood_rate,
    fogler_deactivation_activity,
    A_PDH,
    EA_PDH
)


def test_arrhenius_monotonic_increase():
    """Rate constant must increase monotonically with temperature."""
    t_low = np.array([550.0 + 273.15])
    t_high = np.array([650.0 + 273.15])
    k_low = arrhenius_rate(A_PDH, EA_PDH, t_low)
    k_high = arrhenius_rate(A_PDH, EA_PDH, t_high)
    assert k_high > k_low > 0.0


def test_equilibrium_conversion_bounds():
    """Thermodynamic equilibrium conversion must strictly lie in (0.0, 1.0)."""
    temperatures = np.linspace(500.0, 700.0, 20) + 273.15
    x_eq = equilibrium_conversion_limit(temperatures)
    assert np.all(x_eq >= 0.0)
    assert np.all(x_eq <= 1.0)
    # Endothermic reaction: equilibrium conversion increases with temperature
    assert np.all(np.diff(x_eq) > 0.0)


def test_langmuir_adsorption_exothermic():
    """Adsorption constant must decrease with temperature (exothermic adsorption)."""
    t_low = np.array([520.0 + 273.15])
    t_high = np.array([620.0 + 273.15])
    k_ads_low = langmuir_adsorption_const('C3H8', t_low)
    k_ads_high = langmuir_adsorption_const('C3H8', t_high)
    assert k_ads_low > k_ads_high > 0.0


def test_fogler_deactivation_decay():
    """Catalyst activity must monotonically decay over time."""
    time_series = np.array([0.0, 2.0, 5.0, 10.0, 24.0])
    temp_k = np.full_like(time_series, 580.0 + 273.15)
    c_coke = np.full_like(time_series, 2.5)
    activity = fogler_deactivation_activity(time_series, temp_k, c_coke)
    assert activity[0] == 1.0
    assert np.all(np.diff(activity) <= 0.0)
