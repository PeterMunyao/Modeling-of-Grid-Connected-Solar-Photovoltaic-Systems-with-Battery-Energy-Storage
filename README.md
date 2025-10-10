PV-Battery-Grid Simulation

**A Python-based simulation of solar PV, battery storage, and grid interaction for residential or microgrid applications.**

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
2. **Compute** PV output power (DC → AC).  
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

---

## Requirements

```bash
pip install pandas numpy matplotlib
