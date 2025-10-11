## PV-Battery-Grid Simulation

**A Python-based simulation of solar PV, battery storage and grid interaction for residential or microgrid applications.**

The model operates on time-series irradiance and load data, **prioritizing battery charging before any grid export**.  
Surplus PV power is exported **only when the battery is fully charged**, thereby **maximizing self-consumption**.

## Features

- Imports irradiance data (GHI, DNI, DHI) and computes PV generation.  
- Implements **battery-first logic**: PV charges the battery before exporting to grid.  
- Balances power among **PV, load, battery, and grid** with SOC (State of Charge) constraints.  
- Visualizes **PV generation**, **battery SOC**, and **grid imports/exports**.  
- Supports **full-year** or **daily** performance visualization.


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
A high-resolution (5-minute) Python simulation of **solar PV generation, battery dispatch** at a premises in durban, South Africa. 
The model implements **dynamic load modeling, time-of-use strategies and strategic import/export control** to minimize grid dependency and maximize self-consumption.

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

# 5-Minute High-Resolution Multi-Segment PV-Battery Self-Consumption & Grid Interaction Simulator

This Python project simulates a **residential PV-battery system** with grid interaction, using high-resolution (5-minute) weather and load data. It calculates power flows, battery operation and grid import/export.

## Features

- **Multi-segment PV modeling and simulation:** Accounts for tilt, azimuth, temperature, irradiance, cloud cover, and losses.
- **Realistic residential load profile:** Morning/evening peaks with stochastic variation.
- **Battery energy management:** Maximum self-consumption strategy with SOC limits.
- **Grid interaction:** Tracks import/export separately from battery operation.
- **Energy metrics:** Computes total PV energy, load consumption, grid import/export, self-consumption ratio.
- **Financial impact:** Estimates annual savings based on electricity costs and export revenue.
- **High-quality plots:** Power flow and SOC plots with negative spectrum, suitable for reports.

## MAXIMUM SELF-CONSUMPTION PERFORMANCE SUMMARY

This code implements a maximum self-consumption strategy that prioritizes using solar energy on-site rather than exporting to the grid. The algorithm follows a strict operational sequence: first, the battery discharges to power home loads whenever solar generation is insufficient; second, any excess solar energy charges the battery for later use; and only as a last resort does the system import from the grid. Critically, the system eliminates strategic grid export - unlike previous versions that discharged batteries to the grid during peak hours for revenue, this configuration reserves all stored energy exclusively for self-consumption. The financial benefits are calculated as savings that would otherwise be incurred from consuming grid energy for most of the load profile. The battery parameters are optimized for this purpose with 92% efficiency and conservative state-of-charge limits (15-90%) to maximize lifetime performance. 

# Algorithm: Maximum Self Consumption

The algorithm first imports 5-minute weather data (DNI, GHI, DHI, air temperature, albedo, zenith, azimuth, cloud opacity, humidity, wind speed) and generates a realistic residential load profile where load varies by hour with added Gaussian noise, clipped to [base, peak] values. PV power is computed per segment by calculating the angle of incidence (AOI = arccos(cos(θ_zenith)*cos(θ_tilt) + sin(θ_zenith)*sin(θ_tilt)*cos(φ_sun - φ_tilt))) and total plane-of-array irradiance (POA_total = POA_direct + POA_diffuse + POA_reflected), which is converted to temperature-corrected DC power and then AC power (P_AC = P_DC * inverter_efficiency * losses), summing all segments to obtain total PV output. At each timestep, the power balance is calculated as PV minus load; if power_balance < 0 and battery SOC > SOC_min, the battery discharges (up to min(-power_balance, max_discharge, (SOC - SOC_min)/efficiency/dt)) and supplies the load; elif power_balance > 0 and SOC < SOC_max, the battery charges (up to min(power_balance, max_charge, (SOC_max - SOC)/efficiency/dt)); remaining imbalance is exported to or imported from the grid. Energy metrics are computed as total PV generation, load consumption, grid import/export, self-consumption ratio ((E_load - E_grid_import)/E_load), battery charge/discharge energy, and round-trip efficiency. Financial metrics are calculated as Import_Cost = E_grid_import*3.0, Export_Revenue = E_grid_export*1.5, Net_Cost = Import_Cost - Export_Revenue, and Annual_Savings = E_load*3.0 - Net_Cost. Finally, visualization is performed for daily, weekly, and yearly timelines, plotting PV, load, battery charging/discharging, grid flows, and battery SOC, with negative spectrum representation for discharging and import.

## Financial Impact of PV + Battery System

This section calculates the financial benefit of using the PV + Battery system compared to relying entirely on grid electricity.  

Assumptions:  
- Grid electricity import cost: **R 3.00 per kWh**  
- Grid electricity export revenue: **R 1.50 per kWh**  

Let:

- `E_import` = Total energy imported from the grid (kWh)  
- `E_export` = Total energy exported to the grid (kWh)  
- `E_load` = Total household load energy (kWh)  

### Equations


1. **Cost of imported electricity**:  
   Import Cost = *E_import* × 3.00  

2. **Revenue from exported electricity**:  
   Export Revenue = *E_export* × 1.50  

3. **Net cost of electricity with PV + Battery**:  
   Net Cost = Import Cost − Export Revenue  

4. **Cost without PV + Battery system**:  
   Cost without system = *E_load* × 3.00  

5. **Annual savings**:  
   Savings = Cost without system − Net Cost  

## SCENARIO CASE STUDY RESULTS 

- **Total PV Generation:** 146,413 kWh
- **Total Load Consumption:** 92,706 kWh
- **Grid Import:** 11,526 kWh
- **Grid Export:** 59,919 kWh
- **Self-Consumption Ratio:** 87.6%

- **Battery Performance:**
  - **Energy Charged:** 34,710 kWh
  - **Energy Discharged:** 29,396 kWh
  - **Round-trip Efficiency:** 84.7%

- **Financial Impact (R 3.00/kWh):**
  - **Import Cost:** R 34,578
  - **Export Revenue:** R 89,879
  - **Net Cost:** R -55,302
  - **Annual Savings:** R 333,418


## Requirements

```bash
pip install pandas numpy matplotlib pvlib pytz

```bash
pip install pandas numpy matplotlib.


