# === IMPORT LIBRARIES ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

# Set professional styling for IEEE conference paper
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Garamond']
rcParams['font.size'] = 16
rcParams['font.weight'] = 'normal'
rcParams['axes.labelsize'] = 16
rcParams['axes.titlesize'] = 16
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 16
rcParams['legend.fontsize'] = 14
rcParams['figure.figsize'] = [10, 8]  # Taller format for power + SOC

# === LOAD AND PREPARE WEATHER DATA ===
#file_path = 'csv_-29.815268_30.946439_fixed_23_0_PT5M.csv'
#df = pd.read_csv(file_path)
#df['period_end'] = pd.to_datetime(df['period_end'], utc=True)
#df.set_index('period_end', inplace=True)
#df = df[(df.index >= '2024-01-01') & (df.index < '2025-01-01')]
#df.index = df.index.tz_convert('Africa/Johannesburg')



# USE MANUAL TZ CONVERSION, THE ABOVE TZ IS NOT SHIFTING TIME AS EXPECTED SAST=UTC+2HRS
# === LOAD WEATHER DATA - SHIFT CSV TIME BY +2 HOURS ===
file_path = 'csv_-29.815268_30.946439_fixed_23_0_PT5M.csv'
df = pd.read_csv(file_path)
df['period_end'] = pd.to_datetime(df['period_end'])  # Load with original timezone
df.set_index('period_end', inplace=True)
df = df[(df.index >= '2024-01-01') & (df.index < '2025-01-01')]

# Shift time by +2 hours
df.index = df.index + pd.Timedelta(hours=2)



# Ensure required columns exist
required_columns = ['dni', 'ghi', 'dhi', 'air_temp', 'albedo', 'zenith', 'azimuth',
                    'cloud_opacity', 'relative_humidity', 'wind_speed_10m']
for col in required_columns:
    if col not in df.columns:
        df[col] = 0

# === PV SYSTEM SIMULATION ===
print("Simulating PV system...")
panel_power_max = 600
inverter_efficiency = 0.95
temp_coeff = -0.0045
stc_irradiance = 1000
losses = 0.99

field_segments = [
    {"tilt": 5.6, "azimuth": 319.88214, "num_modules": 32},
    {"tilt": 2.8, "azimuth": 146.61220, "num_modules": 32},
    {"tilt": 5.0, "azimuth": 326.42346, "num_modules": 32},
    {"tilt": 3.0, "azimuth": 315.20587, "num_modules": 32},
    {"tilt": 3.0, "azimuth": 134.65346, "num_modules": 64},
]

df["AC_Power_kW_osm_total"] = 0

for seg in field_segments:
    tilt_rad = np.radians(seg["tilt"])
    az_rad = np.radians(seg["azimuth"])
    num_panels = seg["num_modules"]
    zen_rad = np.radians(df['zenith'])
    sun_az_rad = np.radians(df['azimuth'])

    # PV simulation equations
    aoi = np.degrees(np.arccos(
        np.cos(zen_rad) * np.cos(tilt_rad) +
        np.sin(zen_rad) * np.sin(tilt_rad) * np.cos(sun_az_rad - az_rad)
    ))
    aoi = np.clip(aoi, 0, 90)

    poa_direct = df['dni'] * np.cos(np.radians(aoi)) * (1 - df['cloud_opacity']/100)
    poa_direct = poa_direct.clip(lower=0)
    poa_diffuse = df['dhi'] * (1 + np.cos(tilt_rad)) / 2
    poa_reflected = df['ghi'] * df['albedo'] * (1 - np.cos(tilt_rad)) / 2
    poa_total = poa_direct + poa_diffuse + poa_reflected

    module_temp = 45 + poa_total/1000 * (28 - df['air_temp'])
    
    dc_power = panel_power_max * (1 + temp_coeff*(module_temp - 45))
    dc_power *= poa_total / stc_irradiance
    dc_power *= (1 - 0.002 * df['relative_humidity'])

    ac_power = dc_power * inverter_efficiency
    df["AC_Power_kW_osm_total"] += ac_power * num_panels * losses / 1000

# === REALISTIC LOAD PROFILE ===
print("Creating load profile...")
def create_realistic_load(index, base=5, peak=25):
    """Create realistic residential load profile"""
    load = np.ones(len(index)) * base
    
    for i, timestamp in enumerate(index):
        hour = timestamp.hour
        
        if 6 <= hour < 9:      # Morning peak
            load[i] = base + 0.4*(peak - base)
        elif 9 <= hour < 17:   # Daytime baseline
            load[i] = base + 0.5*(peak - base)
        elif 17 <= hour < 22:  # Evening peak
            load[i] = base + 0.3*(peak - base)
        else:                  # Night
            load[i] = base * 0.3
    
    # Add realistic noise
    load += np.random.normal(0, 1.0, len(load))
    
    return np.clip(load, 5, peak)

df['Home_Load_kW'] = create_realistic_load(df.index)

# === BATTERY SYSTEM ===
battery_capacity_kwh = 150
battery_max_charge_kw = 15
battery_max_discharge_kw = 15
battery_efficiency = 0.92
dt = 5/60  # 5-minute intervals

soc_min = 0.15 * battery_capacity_kwh
soc_max = 0.90 * battery_capacity_kwh
battery_soc = 0.6 * battery_capacity_kwh  # Start 60% charged

# === MAXIMUM SELF-CONSUMPTION STRATEGY ===
print("Running battery simulation...")
grid_power, battery_power, battery_soc_track, operation_mode = [], [], [], []

for i, pv_power in enumerate(df["AC_Power_kW_osm_total"]):
    load = df['Home_Load_kW'].iloc[i]
    current_hour = df.index[i].hour
    
    # Initial power balance
    power_balance = pv_power - load
    battery_power_kw = 0
    mode = "Idle"
    
    # === PRIORITY 1: SUPPLY LOAD FROM BATTERY IF NEEDED ===
    if power_balance < 0 and battery_soc > soc_min:
        power_needed = -power_balance
        available_energy = (battery_soc - soc_min) * battery_efficiency
        max_discharge = min(battery_max_discharge_kw, available_energy / dt)
        discharge_power = min(power_needed, max_discharge)
        
        if discharge_power > 0.5:  # Minimum discharge threshold
            battery_soc -= discharge_power * dt / battery_efficiency
            battery_power_kw = -discharge_power
            power_balance += discharge_power
            mode = "Load Supply"
    
    # === PRIORITY 2: CHARGE BATTERY FROM EXCESS SOLAR ===
    elif power_balance > 0 and battery_soc < soc_max and mode == "Idle":
        available_capacity = (soc_max - battery_soc) / battery_efficiency
        max_charge = min(battery_max_charge_kw, available_capacity / dt)
        charge_power = min(power_balance, max_charge)
        
        if charge_power > 0.5:  # Minimum charge threshold
            battery_soc += charge_power * dt * battery_efficiency
            battery_power_kw = charge_power
            power_balance -= charge_power
            mode = "Charging"
    
    # Final grid power
    grid_power.append(power_balance)
    battery_power.append(battery_power_kw)
    battery_soc_track.append(battery_soc)
    operation_mode.append(mode)

# Add results to dataframe
df['Grid_Power_KW'] = grid_power
df['Battery_Power_kW'] = battery_power
df['Battery_SOC_kWh'] = battery_soc_track
df['Battery_Mode'] = operation_mode

# Create separate columns for better plotting
df['Battery_Charging_kW'] = df['Battery_Power_kW'].clip(lower=0)
df['Battery_Discharging_kW'] = df['Battery_Power_kW'].clip(upper=0).abs()
df['Grid_Export_kW'] = df['Grid_Power_KW'].clip(lower=0)
df['Grid_Import_kW'] = df['Grid_Power_KW'].clip(upper=0).abs()

# === ENERGY CALCULATIONS ===
print("Calculating energy metrics...")
dt_hours = dt
total_pv_energy = df["AC_Power_kW_osm_total"].sum() * dt_hours
total_load_energy = df["Home_Load_kW"].sum() * dt_hours
grid_export_energy = df[df['Grid_Power_KW'] > 0]['Grid_Power_KW'].sum() * dt_hours
grid_import_energy = abs(df[df['Grid_Power_KW'] < 0]['Grid_Power_KW'].sum() * dt_hours)
self_consumption_ratio = (total_load_energy - grid_import_energy) / total_load_energy

# Battery metrics
battery_charge_energy = df[df['Battery_Power_kW'] > 0]['Battery_Power_kW'].sum() * dt_hours
battery_discharge_energy = abs(df[df['Battery_Power_kW'] < 0]['Battery_Power_kW'].sum() * dt_hours)

# === COMPREHENSIVE PLOTTING FUNCTION WITH NEGATIVE SPECTRUM ===
# === COMPREHENSIVE PLOTTING FUNCTION WITH NEGATIVE SPECTRUM ===
def create_power_soc_plot(data, time_period):
    """Create power flow plot with negative spectrum and SOC plot"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [2, 1]})  # Increased height
    
    # Define distinct colors for each variable
    colors = {
        'pv': '#E67E22',           # Orange
        'load': '#C0392B',         # Red
        'battery_charge': '#2980B9',     # Blue
        'battery_discharge': '#8E44AD',  # Purple
        'grid_export': '#27AE60',        # Green
        'grid_import': '#E74C3C',        # Dark Red
        'soc': '#2C3E50',          # Dark Blue
        'soc_min': '#FF6B6B',      # Light Red
        'soc_max': '#4ECDC4'       # Teal
    }
    
    # Plot 1: Power flows with negative spectrum
    # Positive values
    ax1.plot(data.index, data["AC_Power_kW_osm_total"], 
             color=colors['pv'], linewidth=1, label='PV Power', alpha=0.8)
    ax1.plot(data.index, data["Home_Load_kW"], 
             color=colors['load'], linewidth=1, label='Load Profile', alpha=0.8)
    ax1.plot(data.index, data['Battery_Charging_kW'], 
             color=colors['battery_charge'], linewidth=1, label='Battery Charging', alpha=0.8)
    ax1.plot(data.index, data['Grid_Export_kW'], 
             color=colors['grid_export'], linewidth=1, label='PV Export to Grid', alpha=0.8)
    
    # Negative values (discharge and import as negative)
    ax1.plot(data.index, -data['Battery_Discharging_kW'], 
             color=colors['battery_discharge'], linewidth=1, label='Battery Discharging', linestyle='--', alpha=0.8)
    ax1.plot(data.index, -data['Grid_Import_kW'], 
             color=colors['grid_import'], linewidth=1, label='Grid Import', linestyle='--', alpha=0.8)
    
    # Add zero line
    ax1.axhline(y=0, color='black', linewidth=1, alpha=0.5)
    
    ax1.set_ylabel('Power (kW)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Set symmetric y-axis limits based on data range
    max_positive = max(data["AC_Power_kW_osm_total"].max(), data["Home_Load_kW"].max(), 
                       data['Battery_Charging_kW'].max(), data['Grid_Export_kW'].max())
    max_negative = max(data['Battery_Discharging_kW'].max(), data['Grid_Import_kW'].max())
    y_max = max(max_positive, max_negative)
    ax1.set_ylim(-y_max * 1.1, y_max * 1.1)
    
    # Move legend MUCH lower to avoid overlapping - KEY CHANGE
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4),  # Changed from -0.2 to -0.4
               ncol=3, framealpha=0.9, fontsize=11)  # Reduced font size slightly
    
    # Plot 2: Battery State of Charge
    ax2.plot(data.index, data['Battery_SOC_kWh'], 
             color=colors['soc'], linewidth=1, label='Battery SOC', alpha=0.8)
    ax2.axhline(y=soc_max, color=colors['soc_max'], linestyle='--', 
                linewidth=1.5, alpha=0.7, label='Max SOC')
    ax2.axhline(y=soc_min, color=colors['soc_min'], linestyle='--', 
                linewidth=1.5, alpha=0.7, label='Min SOC')
    ax2.set_ylabel('SOC (kWh)', fontweight='bold')
    ax2.set_ylim(0, battery_capacity_kwh * 1.05)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4),  # Changed from -0.2 to -0.4
               ncol=3, framealpha=0.9, fontsize=11)  # Reduced font size slightly
    ax2.grid(True, alpha=0.3)
    
    # X-axis formatting based on time period
    for ax in [ax1, ax2]:
        if time_period == 'yearly':
            # For yearly plot with 5-min data, use monthly ticks
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            # Add minor ticks for weeks
            ax.xaxis.set_minor_locator(mdates.WeekdayLocator(interval=2))
        elif time_period == 'weekly':
            # For weekly plot with 5-min data, use daily ticks
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%a\n%d %b'))
            # Add minor ticks for every 6 hours
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
        elif time_period == 'daily':
            # For daily plot with 5-min data, use hourly ticks
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            # Add minor ticks for every hour
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
    
    # Remove titles as requested
    ax1.set_title('')
    ax2.set_title('')
    
    # Adjust layout with more padding - KEY CHANGE
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Added bottom padding
    return fig

# === GENERATE SPECIFIED TIMELINE PLOTS IN 5-MINUTE TIMESTEPS ===
print("Generating plots for specified timelines in 5-minute timesteps...")

# 1. Yearly 2024 Plot - Original 5-minute data (will be dense but shows patterns)
print("Creating yearly 2024 plot with 5-minute data...")
yearly_data = df.copy()  # Use all original 5-minute data
fig_yearly = create_power_soc_plot(yearly_data, 'yearly')
plt.savefig('yearly_5min_power_soc.pdf', dpi=300, bbox_inches='tight')
plt.show()

# 2. January 7 days - Original 5-minute data
print("Creating January 7 days plot with 5-minute data...")
jan_data = df[(df.index >= '2024-01-01') & (df.index < '2024-01-08')].copy()
fig_jan = create_power_soc_plot(jan_data, 'weekly')
plt.savefig('january_7days_5min_power_soc.pdf', dpi=300, bbox_inches='tight')
plt.show()

# 3. July 7 days - Original 5-minute data
print("Creating July 7 days plot with 5-minute data...")
jul_data = df[(df.index >= '2024-07-01') & (df.index < '2024-07-08')].copy()
fig_jul = create_power_soc_plot(jul_data, 'weekly')
plt.savefig('july_7days_5min_power_soc.pdf', dpi=300, bbox_inches='tight')
plt.show()

# 4. August 7 days - Original 5-minute data
print("Creating August 7 days plot with 5-minute data...")
aug_data = df[(df.index >= '2024-08-01') & (df.index < '2024-08-08')].copy()
fig_aug = create_power_soc_plot(aug_data, 'weekly')
plt.savefig('august_7days_5min_power_soc.pdf', dpi=300, bbox_inches='tight')
plt.show()

# 5. December 7 days - Original 5-minute data
print("Creating December 7 days plot with 5-minute data...")
dec_data = df[(df.index >= '2024-12-01') & (df.index < '2024-12-08')].copy()
fig_dec = create_power_soc_plot(dec_data, 'weekly')
plt.savefig('december_7days_5min_power_soc.pdf', dpi=300, bbox_inches='tight')
plt.show()

# === PERFORMANCE SUMMARY ===
print("\n" + "="*70)
print("MAXIMUM SELF-CONSUMPTION PERFORMANCE SUMMARY")
print("="*70)
print(f"Total PV Generation: {total_pv_energy:,.0f} kWh")
print(f"Total Load Consumption: {total_load_energy:,.0f} kWh")
print(f"Grid Import: {grid_import_energy:,.0f} kWh")
print(f"Grid Export: {grid_export_energy:,.0f} kWh")
print(f"Self-Consumption Ratio: {self_consumption_ratio:.1%}")
print(f"\nBattery Performance:")
print(f"  Energy Charged: {battery_charge_energy:,.0f} kWh")
print(f"  Energy Discharged: {battery_discharge_energy:,.0f} kWh")
print(f"  Round-trip Efficiency: {(battery_discharge_energy/battery_charge_energy*100 if battery_charge_energy > 0 else 0):.1f}%")

# Calculate cost savings (assuming R 3.00/kWh)
import_cost = grid_import_energy * 3.00
export_revenue = grid_export_energy * 1.50
net_cost = import_cost - export_revenue
cost_without_system = total_load_energy * 3.00
savings = cost_without_system - net_cost

print(f"\nFinancial Impact (R 3.00/kWh):")
print(f"  Import Cost: R {import_cost:,.0f}")
print(f"  Export Revenue: R {export_revenue:,.0f}") 
print(f"  Net Cost: R {net_cost:,.0f}")
print(f"  Annual Savings: R {savings:,.0f}")

print(f"\n5-minute resolution Power + SOC plots saved:")
print("  - yearly_5min_power_soc.pdf")
print("  - january_7days_5min_power_soc.pdf")
print("  - july_7days_5min_power_soc.pdf")
print("  - august_7days_5min_power_soc.pdf")
print("  - december_7days_5min_power_soc.pdf")

#----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#----CODE FOR GRAPHS UPDATED-----------------------------------------------------------

# === IMPORT LIBRARIES ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

# Set Garamond bold font size 18 for all labels
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Garamond']
rcParams['font.size'] = 18
rcParams['font.weight'] = 'bold'
rcParams['axes.labelsize'] = 18
rcParams['axes.titlesize'] = 18
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16
rcParams['legend.fontsize'] = 14

# === LOAD WEATHER DATA - SHIFT CSV TIME BY +2 HOURS ===
file_path = 'csv_-29.815268_30.946439_fixed_23_0_PT5M.csv'
df = pd.read_csv(file_path)
df['period_end'] = pd.to_datetime(df['period_end'])  # Load with original timezone
df.set_index('period_end', inplace=True)
df = df[(df.index >= '2024-01-01') & (df.index < '2025-01-01')]

# Shift time by +2 hours
df.index = df.index + pd.Timedelta(hours=2)

# Ensure required columns exist
required_columns = ['dni', 'ghi', 'dhi', 'air_temp', 'albedo', 'zenith', 'azimuth',
                    'cloud_opacity', 'relative_humidity', 'wind_speed_10m']
for col in required_columns:
    if col not in df.columns:
        df[col] = 0

# === PV SYSTEM SIMULATION ===
print("Simulating PV system...")
panel_power_max = 600
inverter_efficiency = 0.95
temp_coeff = -0.0045
stc_irradiance = 1000
losses = 0.99

field_segments = [
    {"tilt": 5.6, "azimuth": 319.88214, "num_modules": 32},
    {"tilt": 2.8, "azimuth": 146.61220, "num_modules": 32},
    {"tilt": 5.0, "azimuth": 326.42346, "num_modules": 32},
    {"tilt": 3.0, "azimuth": 315.20587, "num_modules": 32},
    {"tilt": 3.0, "azimuth": 134.65346, "num_modules": 64},
]

df["AC_Power_kW_osm_total"] = 0

for seg in field_segments:
    tilt_rad = np.radians(seg["tilt"])
    az_rad = np.radians(seg["azimuth"])
    num_panels = seg["num_modules"]
    zen_rad = np.radians(df['zenith'])
    sun_az_rad = np.radians(df['azimuth'])

    # PV simulation equations
    aoi = np.degrees(np.arccos(
        np.cos(zen_rad) * np.cos(tilt_rad) +
        np.sin(zen_rad) * np.sin(tilt_rad) * np.cos(sun_az_rad - az_rad)
    ))
    aoi = np.clip(aoi, 0, 90)

    poa_direct = df['dni'] * np.cos(np.radians(aoi)) * (1 - df['cloud_opacity']/100)
    poa_direct = poa_direct.clip(lower=0)
    poa_diffuse = df['dhi'] * (1 + np.cos(tilt_rad)) / 2
    poa_reflected = df['ghi'] * df['albedo'] * (1 - np.cos(tilt_rad)) / 2
    poa_total = poa_direct + poa_diffuse + poa_reflected

    module_temp = 45 + poa_total/1000 * (28 - df['air_temp'])
    
    dc_power = panel_power_max * (1 + temp_coeff*(module_temp - 45))
    dc_power *= poa_total / stc_irradiance
    dc_power *= (1 - 0.002 * df['relative_humidity'])

    ac_power = dc_power * inverter_efficiency
    df["AC_Power_kW_osm_total"] += ac_power * num_panels * losses / 1000

# === REALISTIC LOAD PROFILE ===
print("Creating load profile...")
def create_realistic_load(index, base=5, peak=25):
    """Create realistic residential load profile"""
    load = np.ones(len(index)) * base
    
    for i, timestamp in enumerate(index):
        hour = timestamp.hour
        
        if 6 <= hour < 9:      # Morning peak
            load[i] = base + 0.4*(peak - base)
        elif 9 <= hour < 17:   # Daytime baseline
            load[i] = base + 0.5*(peak - base)
        elif 17 <= hour < 22:  # Evening peak
            load[i] = base + 0.3*(peak - base)
        else:                  # Night
            load[i] = base * 0.3
    
    # Add realistic noise
    load += np.random.normal(0, 1.0, len(load))
    
    return np.clip(load, 5, peak)

df['Home_Load_kW'] = create_realistic_load(df.index)

# === BATTERY SYSTEM ===
battery_capacity_kwh = 150
battery_max_charge_kw = 15
battery_max_discharge_kw = 15
battery_efficiency = 0.92
dt = 5/60  # 5-minute intervals

soc_min = 0.15 * battery_capacity_kwh
soc_max = 0.90 * battery_capacity_kwh
battery_soc = 0.6 * battery_capacity_kwh  # Start 60% charged

# === MAXIMUM SELF-CONSUMPTION STRATEGY ===
print("Running battery simulation...")
grid_power, battery_power, battery_soc_track, operation_mode = [], [], [], []

for i, pv_power in enumerate(df["AC_Power_kW_osm_total"]):
    load = df['Home_Load_kW'].iloc[i]
    current_hour = df.index[i].hour
    
    # Initial power balance
    power_balance = pv_power - load
    battery_power_kw = 0
    mode = "Idle"
    
    # === PRIORITY 1: SUPPLY LOAD FROM BATTERY IF NEEDED ===
    if power_balance < 0 and battery_soc > soc_min:
        power_needed = -power_balance
        available_energy = (battery_soc - soc_min) * battery_efficiency
        max_discharge = min(battery_max_discharge_kw, available_energy / dt)
        discharge_power = min(power_needed, max_discharge)
        
        if discharge_power > 0.5:  # Minimum discharge threshold
            battery_soc -= discharge_power * dt / battery_efficiency
            battery_power_kw = -discharge_power
            power_balance += discharge_power
            mode = "Load Supply"
    
    # === PRIORITY 2: CHARGE BATTERY FROM EXCESS SOLAR ===
    elif power_balance > 0 and battery_soc < soc_max and mode == "Idle":
        available_capacity = (soc_max - battery_soc) / battery_efficiency
        max_charge = min(battery_max_charge_kw, available_capacity / dt)
        charge_power = min(power_balance, max_charge)
        
        if charge_power > 0.5:  # Minimum charge threshold
            battery_soc += charge_power * dt * battery_efficiency
            battery_power_kw = charge_power
            power_balance -= charge_power
            mode = "Charging"
    
    # Final grid power
    grid_power.append(power_balance)
    battery_power.append(battery_power_kw)
    battery_soc_track.append(battery_soc)
    operation_mode.append(mode)

# Add results to dataframe
df['Grid_Power_KW'] = grid_power
df['Battery_Power_kW'] = battery_power
df['Battery_SOC_kWh'] = battery_soc_track
df['Battery_Mode'] = operation_mode

# Create separate columns for better plotting
df['Battery_Charging_kW'] = df['Battery_Power_kW'].clip(lower=0)
df['Battery_Discharging_kW'] = df['Battery_Power_kW'].clip(upper=0).abs()
df['Grid_Export_kW'] = df['Grid_Power_KW'].clip(lower=0)
df['Grid_Import_kW'] = df['Grid_Power_KW'].clip(upper=0).abs()

# === ENERGY CALCULATIONS ===
print("Calculating energy metrics...")
dt_hours = dt
total_pv_energy = df["AC_Power_kW_osm_total"].sum() * dt_hours
total_load_energy = df["Home_Load_kW"].sum() * dt_hours
grid_export_energy = df[df['Grid_Power_KW'] > 0]['Grid_Power_KW'].sum() * dt_hours
grid_import_energy = abs(df[df['Grid_Power_KW'] < 0]['Grid_Power_KW'].sum() * dt_hours)
self_consumption_ratio = (total_load_energy - grid_import_energy) / total_load_energy

# Battery metrics
battery_charge_energy = df[df['Battery_Power_kW'] > 0]['Battery_Power_kW'].sum() * dt_hours
battery_discharge_energy = abs(df[df['Battery_Power_kW'] < 0]['Battery_Power_kW'].sum() * dt_hours)

# === STRATEGIC PLOT FUNCTION IN YOUR SPECIFIED STYLE ===
def create_strategic_plot(data, time_period='daily', filename='strategic_plot'):
    """Professional plot with strategic energy management visualization in your specified style."""
    
    # Create figure with requested size
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 8.5))
    fig.subplots_adjust(hspace=0.4)
    
    # Define strategic color scheme (your exact colors)
    colors = {
        'pv': '#FF6B00',           # Bright orange for PV
        'load': '#C00000',         # Dark red for load
        'battery_charge': '#0047AB', # Blue for battery charging
        'battery_discharge': '#1E90FF', # Light blue for battery discharging
        'export': '#38761D',       # Green for export
        'import': '#990000',       # Dark red for import
        'soc': '#1F4E79',          # Navy blue for SOC
        'import_fill': '#FFE6E6',  # Very light red for import area
        'export_fill': '#E6FFE6',  # Very light green for export area
        'pv_charging': '#FFA500',  # Orange for PV charging periods
        'strategic_export': '#32CD32', # Lime green for strategic export periods
    }
    
    # --- TOP PLOT: POWER FLOWS ---
    
    # Calculate consistent y-axis limits - LIMIT NEGATIVE AXIS TO -30 kW
    all_power_data = np.concatenate([
        data["AC_Power_kW_osm_total"].values,
        data["Home_Load_kW"].values, 
        data["Battery_Power_kW"].values,
        data['Grid_Export_kW'].values,
        data['Grid_Import_kW'].values
    ])
    
    # Set negative limit to -30 kW, positive limit dynamic with 10% margin
    negative_limit = -30
    positive_max = np.max(all_power_data) * 1.1
    power_ylim = [negative_limit, positive_max]
    
    # Plot power flows
    ax1.plot(data.index, data["AC_Power_kW_osm_total"], 
             label="PV Power", color=colors['pv'], linewidth=1.5)
    ax1.plot(data.index, data["Home_Load_kW"], 
             label="Load Power", color=colors['load'], linewidth=1.5)
    
    # Battery power
    ax1.plot(data.index, data['Battery_Charging_kW'], 
             label="Battery Charging", color=colors['battery_charge'], linewidth=1.5)
    ax1.plot(data.index, -data['Battery_Discharging_kW'], 
             label="Battery Discharging", color=colors['battery_discharge'], linewidth=1.5)
    
    # Grid power with area fills
    ax1.fill_between(data.index, data['Grid_Export_kW'], alpha=0.3, 
                     color=colors['export_fill'], label='Export Area')
    ax1.fill_between(data.index, -data['Grid_Import_kW'], alpha=0.3, 
                     color=colors['import_fill'], label='Import Area')
    
    ax1.plot(data.index, data['Grid_Export_kW'], 
             label="Export Power", color=colors['export'], linewidth=1.0, linestyle='--')
    ax1.plot(data.index, -data['Grid_Import_kW'], 
             label="Import Power", color=colors['import'], linewidth=1.0, linestyle='--')
    
    ax1.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
    
    # SET Y-AXIS LIMITS WITH -30 kW NEGATIVE LIMIT
    ax1.set_ylim(power_ylim)
    ax1.set_ylabel("Power [kW]", fontname='Garamond', fontsize=16, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # X-axis formatting - START FROM MIDNIGHT, EVERY 6 HOURS
    ax1.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6)))  # 0, 6, 12, 18
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center', fontname='Garamond', fontsize=14)
    
    # Legend for top plot
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), 
               ncol=4, framealpha=0.9, fontsize=11, fancybox=True, shadow=True)
    
    # --- BOTTOM PLOT: BATTERY SOC ---
    
    ax2.plot(data.index, data["Battery_SOC_kWh"], 
             label="Battery SOC", color=colors['soc'], linewidth=2.0)
    ax2.axhline(soc_max, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='Max SOC')
    ax2.axhline(soc_min, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='Min SOC')
    
    ax2.set_ylim(0, battery_capacity_kwh * 1.05)
    ax2.set_ylabel("Battery SOC [kWh]", fontname='Garamond', fontsize=16, fontweight='bold')
    ax2.set_xlabel("Time", fontname='Garamond', fontsize=16, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # X-axis formatting - SAME AS TOP (START FROM MIDNIGHT, EVERY 6 HOURS)
    ax2.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6)))  # 0, 6, 12, 18
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center', fontname='Garamond', fontsize=14)
    
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), 
               ncol=3, framealpha=0.9, fontsize=11, fancybox=True, shadow=True)
    
    # Ensure same x-axis limits
    xlim = (data.index.min(), data.index.max())
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    
    plt.tight_layout()
    plt.savefig(f'{filename}.pdf', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='pdf')
    
    return fig

# === GENERATE PLOTS FOR DAYS 4-7 ONLY ===
print("Generating strategic energy management plots for days 4-7...")

# Define months to plot
months_to_plot = [
    ('January', '2024-01-04', '2024-01-08'),
    ('July', '2024-07-04', '2024-07-08'), 
    ('December', '2024-12-04', '2024-12-08')
]

for month_name, start_date, end_date in months_to_plot:
    plot_data = df[(df.index >= start_date) & (df.index < end_date)]
    if len(plot_data) > 0:
        filename = f"{month_name.lower()}_days_4_7_strategic_peter"
        print(f"Creating {month_name} days 4-7 plot...")
        fig = create_strategic_plot(plot_data, 'daily', filename)
        plt.savefig(f"{filename}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        plt.show()
        plt.close(fig)

# === STRATEGIC PERFORMANCE ANALYSIS ===
print("\n" + "="*60)
print("STRATEGIC ENERGY MANAGEMENT PERFORMANCE")
print("="*60)
print(f"Total PV Energy: {total_pv_energy:,.0f} kWh")
print(f"Total Load Energy: {total_load_energy:,.0f} kWh")
print(f"Grid Import: {grid_import_energy:,.0f} kWh")
print(f"Grid Export: {grid_export_energy:,.0f} kWh")
print(f"Self-Consumption Ratio: {self_consumption_ratio:.1%}")

print(f"\nBattery Charging from PV: {battery_charge_energy:.1f} kWh")
print(f"Battery Discharging to Load: {battery_discharge_energy:.1f} kWh")

# Battery mode analysis
mode_counts = df['Battery_Mode'].value_counts()
print(f"\nBattery Operation Modes:")
for mode, count in mode_counts.items():
    percentage = (count / len(df)) * 100
    hours = count * dt
    print(f"  {mode}: {count:,} intervals ({percentage:.1f}%) - {hours:.1f} hours")

print("\nAll strategic energy management plots saved as PDF files.")
print("Plots generated for:")
print("  - January days 4-7")
print("  - July days 4-7") 
print("  - December days 4-7")




