A Python-based simulation of solar PV, battery storage and grid interaction for residential or microgrid applications.
The model operates on time-series irradiance and load data, prioritizing battery charging before any grid export.
Surplus PV power is exported only when the battery is fully charged thus maximizing self-consumption.

Features

Imports irradiance data (GHI, DNI, DHI) and computes PV generation.

Implements battery-first logic: PV charges the battery before exporting.

Balances power among PV, load, battery, and grid with SOC constraints.

Visualizes PV generation, battery SOC, and grid imports/exports.

Supports full-year or daily performance visualization.

Algorithm Summary

Load weather and timestamped PV data.

Compute PV output power (DC → AC).

Compare PV to load → calculate net power.

If PV > load:

Priority 1: Charge battery (until SOC_max).

Priority 2: Export remaining PV to grid (only if battery full).

If PV < load:

Discharge battery (until SOC_min).

Import remaining power from grid.

Update SOC based on efficiency and record system state.

Plot PV, load, battery, and grid power dynamics.

Outputs

PV power (kW)

Battery SOC (%)

Grid import/export (kWh)

Energy balance and self-consumption ratio

Requirements
pip install pandas numpy matplotlib
