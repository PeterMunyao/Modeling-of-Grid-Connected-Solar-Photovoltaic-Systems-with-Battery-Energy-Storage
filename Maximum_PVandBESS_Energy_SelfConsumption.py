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
