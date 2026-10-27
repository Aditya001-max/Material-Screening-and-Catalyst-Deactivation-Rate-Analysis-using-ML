# Dataset Architecture & Schema

Author: **Aditya001-max** (`kaushikadityaraj001@gmail.com`)  
GitHub: [https://github.com/Aditya001-max](https://github.com/Aditya001-max)

## 1. Overview
This directory contains experimental and benchmark screening datasets for **Propane Dehydrogenation (PDH)** catalyst optimization and deactivation rate modeling.

Reaction stoichiometry:
$$\text{C}_3\text{H}_8 \rightleftharpoons \text{C}_3\text{H}_6 + \text{H}_2 \quad (\Delta H^\circ_{298} = +124\text{ kJ/mol})$$

## 2. Feature Definitions & Units

| Variable Name | Type | Unit | Description |
| :--- | :---: | :---: | :--- |
| `time_of_reaction_h` | Continuous | $\text{hours}$ | Time-on-stream reaction duration |
| `temperature_C` | Continuous | $^\circ\text{C}$ | Reactor operating temperature ($500 - 650^\circ\text{C}$) |
| `Propane_H2_ratio` | Continuous | Molar ratio | Feed ratio of propane to co-fed hydrogen |
| `Partial_pressure_propane_bar` | Continuous | $\text{bar}$ | Partial pressure of propane at reactor inlet |
| `Surface_BET_m2_g` | Continuous | $\text{m}^2/\text{g}$ | Specific BET surface area of the catalyst |
| `Pt_loading_wt%` | Continuous | $\text{wt}\%$ | Active metal platinum loading |
| `Pore_size_nm` | Continuous | $\text{nm}$ | Average pore diameter |
| `Pore_volume_cm3_g` | Continuous | $\text{cm}^3/\text{g}$ | Total cumulative pore volume |
| `Reduction_Temperature_TPR_C` | Continuous | $^\circ\text{C}$ | Catalyst reduction temperature from $\text{H}_2$-TPR |
| `Catalyst` | Categorical | - | Support carrier (`Pt/Al2O3`, `Pt-Sn/Al2O3`, `Pt-In/Al2O3`, etc.) |
| `Dopants_Pt-M` | Categorical | - | Metallic promoter dopant (`Sn`, `In`, `Ga`, `Ce`, `K`, `None`) |

## 3. Multi-Output Target Variables

| Target Variable | Target Name | Physical Ceiling / Law |
| :--- | :--- | :--- |
| **Conversion (%)** | `Conversion_%` | Bounded by van't Hoff equilibrium $X_{\text{eq}}(T)$ |
| **Selectivity (%)** | `Selectivity_%` | Moles of $\text{C}_3\text{H}_6$ produced per mole of $\text{C}_3\text{H}_8$ converted |
| **Propene Yield (%)** | `Propene_yield%` | Strictly governed by stoichiometry: $\frac{\text{Conversion} \times \text{Selectivity}}{100}$ |
| **Coke Deposition (wt%)** | `Coke_deposition_wt%` | Surface fouling accumulation over time |

## 4. Benchmark Dataset Generation
To re-generate or augment the benchmark dataset:
```bash
python data/generate_sample_data.py
```
This produces `pdh_catalyst_data_sample.csv` (713 samples) conforming to thermodynamic and kinetic laws.
