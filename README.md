## PV-Battery-Grid Simulation

**A Python-based simulation of solar PV, battery storage and grid interaction for residential or microgrid applications.**

The model operates on time-series irradiance and load data, **prioritizing battery charging before any grid export**.  
Surplus PV power is exported **only when the battery is fully charged**, thereby **maximizing self-consumption**.

## Features

- 📥 Imports irradiance data (GHI, DNI, DHI) and computes PV generation.  
- ⚡ Implements **battery-first logic**: PV charges the battery before exporting to grid.  
- 🔁 Balances power among **PV, load, battery, and grid** with SOC (State of Charge) constraints.  
- 📊 Visualizes **PV generation**, **battery SOC**, and **grid imports/exports**.  
- 🗓️ Supports **full-year** or **daily** performance visualization.


## Algorithm Summary

1. **Load** weather and timestamped PV data.  
2. **Compute** PV output power (DC → AC) using OSM-MEPS code.  
3. **Compare** PV generation to load → calculate net power.  
4. **Decision Logic:**  
   - If **PV > load**:  
     - **Priority 1:** Charge battery (until `SOC_max`).  
     - **Priority 2:** Export remaining PV to grid (**only if battery is full**).  
   - If **PV < load**:  
     - **Discharge** battery (until `SOC_min`).  
     - **Import** remaining power from grid.  
5. **Update SOC** based on charge/discharge efficiency and record system state.  
6. **Plot results** showing PV, load, battery, and grid power dynamics.


## Outputs

- **PV power** (kW)  
- **Battery SOC** (%)  
- **Grid import/export** (kWh)  
- **Energy balance** and **self-consumption ratio**

## PV-Battery-Grid Energy Management Simulator

## Description

A high-resolution (5-minute) simulation of PV generation, battery dispatch, and smart grid interaction under South African 2024 conditions. Implements dynamic load modeling, time-of-use strategy, and strategic import/export control to minimize grid dependency and maximize self-consumption.

## Key Features

- Multi-configuration PV modeling (tilt, azimuth, derating)
- Realistic load profile with day/night/peak patterns
- Battery operation under SOC and power constraints
- Strategic export control.
  
## Algorithm Overview

1. **Input Data**: Import 5-min irradiance and weather CSV (2024)
2. **PV Model**: Compute AOI, POA, temperature-corrected DC → AC power
3. **Load Modeling**: Dynamic + noise with min/max limits
4. **Battery Management**:
   - Charge when PV > load
   - Discharge when PV < load
   - Strategic export during evening peak hours
5. **Grid Interaction**: Import/export according to SOC and thresholds
6. **Visualization**: Plot daily/weekly/yearly PV, load, SOC, and grid flow


## Outputs

- PV power, load demand, SOC, grid exchange
- Energy KPIs (PV, import, export, load, SCR)


# PV-Battery-Grid Energy Management Simulator

## Description
A high-resolution (5-minute) Python simulation of **solar PV generation, battery dispatch, and smart grid interaction** under South African 2024 conditions.  
The model implements **dynamic load modeling, time-of-use strategies, and strategic import/export control** to minimize grid dependency and maximize self-consumption.

---

## Key Features
- Multi-surface PV field modeling (tilt, azimuth, derating).
- Realistic load profile with **day/night/peak patterns**.
- Battery operation under SOC and power constraints.
- **Strategic export control** between 18:00–21:00.
- Performance KPIs: PV energy, import/export, SOC, SCR.
- Publication-ready Garamond plots (PDF output).

---

## Algorithm Overview
1. **Input Data**: Import 5-min irradiance and weather CSV (2024).
2. **PV Model**: Compute AOI, POA, temperature-corrected DC → AC power.
3. **Load Modeling**: Dynamic + noise with min/max limits.
4. **Battery Management**:
   - Charge when PV > load.
   - Discharge when PV < load.
   - Strategic export during evening peak hours (6pm-9pm).
5. **Grid Interaction**: Import/export according to SOC and thresholds.
6. **Visualization**: Plot daily/weekly/yearly PV, load, SOC, and grid flow.

---

## Outputs
- PV power, load demand, SOC, grid exchange
- Energy KPIs (PV, import, export, load, SCR)
- 14 auto-saved plots: daily to yearly views
- “Strategic hours” shaded visualization

---

## Requirements

```bash
pip install pandas numpy matplotlib pvlib pytz

```bash
pip install pandas numpy matplotlib.

