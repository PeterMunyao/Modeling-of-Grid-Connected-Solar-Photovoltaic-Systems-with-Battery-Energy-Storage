#======================================================================
# OPTIMAL SYSTEM SIZING AND ENERGY MANAGEMENT ANALYSIS
#======================================================================

#RECOMMENDED OPTIMAL SYSTEM SIZING:
#Optimal Battery Capacity: 138 kWh
#Optimal Inverter Size: 106 kW
#Battery Amp-Hours (48V system): 2868 Ah
#PV System Size: 115.2 kW (current)

#SCENARIO ANALYSIS:

#No Import Scenario:
  #Total Annual Load: 146,363 kWh
 # Average Load: 16.7 kW
 # Peak Load: 105.7 kW

#With Export Scenario:
  #Total Annual Load: 82,368 kWh
  #Average Load: 9.4 kW
  #Peak Load: 20.0 kW
  #Evening Export (6-9pm): 0 kWh

#Optimal Shifting Scenario:
  #Total Annual Load: 134,535 kWh
  #Average Load: 15.4 kW
  #Peak Load: 22.5 kW




# === IMPORT LIBRARIES ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

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

# === LOAD WEATHER DATA ===
file_path = 'csv_-29.815268_30.946439_fixed_23_0_PT5M.csv'
df = pd.read_csv(file_path)
df['period_end'] = pd.to_datetime(df['period_end'], utc=True)
df.set_index('period_end', inplace=True)
df = df[(df.index >= '2024-01-01') & (df.index < '2025-01-01')]
df.index = df.index.tz_convert('Africa/Johannesburg')

# === ENSURE REQUIRED COLUMNS EXIST ===
required_columns = ['dni', 'ghi', 'dhi', 'air_temp', 'albedo', 'zenith', 'azimuth',
                    'cloud_opacity', 'relative_humidity', 'wind_speed_10m']
for col in required_columns:
    if col not in df.columns:
        df[col] = 0

# === PV SYSTEM PARAMETERS ===
panel_power_max = 600        # W per module at STC
inverter_efficiency = 0.95   # η_inv
temp_coeff = -0.0045         # β_temp (°C⁻¹)
stc_irradiance = 1000        # G_STC (W/m²)
losses = 0.99                # System losses

field_segments = [
    {"tilt": 5.6, "azimuth": 319.88214, "num_modules": 32},
    {"tilt": 2.8, "azimuth": 146.61220, "num_modules": 32},
    {"tilt": 5.0, "azimuth": 326.42346, "num_modules": 32},
    {"tilt": 3.0, "azimuth": 315.20587, "num_modules": 32},
    {"tilt": 3.0, "azimuth": 134.65346, "num_modules": 64},
]

df["AC_Power_kW_osm_total"] = 0

# === OSM-MEPS PV SIMULATION EQUATIONS ===
for seg in field_segments:
    tilt_rad = np.radians(seg["tilt"])
    az_rad = np.radians(seg["azimuth"])
    num_panels = seg["num_modules"]
    zen_rad = np.radians(df['zenith'])
    sun_az_rad = np.radians(df['azimuth'])

    # Equation 1: Angle of Incidence (AOI) calculation
    aoi = np.degrees(np.arccos(
        np.cos(zen_rad) * np.cos(tilt_rad) +
        np.sin(zen_rad) * np.sin(tilt_rad) * np.cos(sun_az_rad - az_rad)
    ))
    aoi = np.clip(aoi, 0, 90)

    # Equation 2-5: Plane of Array (POA) Irradiance
    poa_direct = df['dni'] * np.cos(np.radians(aoi)) * (1 - df['cloud_opacity']/100)
    poa_direct = poa_direct.clip(lower=0)
    poa_diffuse = df['dhi'] * (1 + np.cos(tilt_rad)) / 2
    poa_reflected = df['ghi'] * df['albedo'] * (1 - np.cos(tilt_rad)) / 2
    poa_total = poa_direct + poa_diffuse + poa_reflected

    # Equation 6: Module temperature model
    module_temp = 45 + poa_total/1000 * (28 - df['air_temp'])
    
    # Equation 7: DC power with temperature correction
    dc_power = panel_power_max * (1 + temp_coeff*(module_temp - 45))
    dc_power *= poa_total / stc_irradiance
    
    # Equation 8: Humidity derating
    dc_power *= (1 - 0.002 * df['relative_humidity'])

    # Equation 9: AC power conversion
    ac_power = dc_power * inverter_efficiency
    
    # Equation 10: Scale by number of panels
    df["AC_Power_kW_osm_total"] += ac_power * num_panels * losses / 1000

# === OPTIMIZATION ALGORITHMS ===
def calculate_optimal_load_profile(pv_power, battery_capacity_kwh, battery_power_kw, 
                                 peak_export_hours=[18, 19, 20], dt=5/60):
    """
    Calculate maximum load that can be supplied without imports and 
    load that ensures export during peak hours.
    """
    
    # Scenario 1: Maximum load without imports
    battery_soc = 0.5 * battery_capacity_kwh
    soc_min = 0.10 * battery_capacity_kwh
    soc_max = 0.95 * battery_capacity_kwh
    battery_efficiency = 0.90
    
    max_load_no_import = np.zeros(len(pv_power))
    
    for i in range(len(pv_power)):
        available_power = pv_power[i]
        
        # Add battery discharge if needed and available
        if available_power < 0 and battery_soc > soc_min:
            discharge_possible = min(-available_power, battery_power_kw)
            energy_available = (battery_soc - soc_min) * battery_efficiency
            discharge_power = min(discharge_possible, energy_available / dt)
            available_power += discharge_power
            battery_soc -= discharge_power * dt / battery_efficiency
        
        # Charge battery if excess PV
        elif available_power > 0 and battery_soc < soc_max:
            charge_possible = min(available_power, battery_power_kw)
            energy_needed = (soc_max - battery_soc) / battery_efficiency
            charge_power = min(charge_possible, energy_needed / dt)
            available_power -= charge_power
            battery_soc += charge_power * dt * battery_efficiency
        
        max_load_no_import[i] = max(0, available_power)
    
    # Scenario 2: Load that ensures export during peak hours
    battery_soc = 0.8 * battery_capacity_kwh  # Start with high SOC for export
    load_with_export = np.zeros(len(pv_power))
    target_export = 10  # kW target export during peak hours
    
    for i in range(len(pv_power)):
        current_hour = pv_power.index[i].hour
        available_power = pv_power[i]
        
        if current_hour in peak_export_hours:
            # CORRECTED LOGIC: Supply load first, then export surplus
            base_load = 15  # Define a reasonable base load
            
            # Step 1: PV supplies load first
            load_supply_from_pv = min(available_power, base_load)
            remaining_load = base_load - load_supply_from_pv
            excess_pv = max(available_power - base_load, 0)
            
            # Step 2: Battery supplies remaining load
            if remaining_load > 0 and battery_soc > soc_min:
                battery_load_supply = min(remaining_load, battery_power_kw)
                energy_available = (battery_soc - soc_min) * battery_efficiency
                battery_load_supply = min(battery_load_supply, energy_available / dt)
                remaining_load -= battery_load_supply
                battery_soc -= battery_load_supply * dt / battery_efficiency
            else:
                battery_load_supply = 0
            
            # Step 3: Battery exports surplus after load is supplied
            available_for_export = battery_power_kw - battery_load_supply
            if available_for_export > 0 and battery_soc > soc_min:
                energy_available = (battery_soc - soc_min) * battery_efficiency
                battery_export = min(available_for_export, energy_available / dt)
                battery_export = min(battery_export, target_export - excess_pv)
                battery_soc -= battery_export * dt / battery_efficiency
            else:
                battery_export = 0
            
            total_export = excess_pv + battery_export
            load_with_export[i] = base_load - remaining_load  # Actual load supplied
            
        else:
            # Normal operation - charge battery
            load_with_export[i] = min(available_power, 20)  # Reasonable load limit
            excess_for_charging = max(available_power - load_with_export[i], 0)
            
            if excess_for_charging > 0 and battery_soc < soc_max:
                charge_possible = min(excess_for_charging, battery_power_kw)
                energy_needed = (soc_max - battery_soc) / battery_efficiency
                charge_power = min(charge_possible, energy_needed / dt)
                battery_soc += charge_power * dt * battery_efficiency
    
    return max_load_no_import, load_with_export

def create_optimal_load_shifting(pv_power, base_load=15, peak_hours=[18, 19, 20, 21]):
    """
    Create optimal load profile with peak shaving and load shifting.
    """
    optimal_load = np.ones(len(pv_power)) * base_load
    
    for i, timestamp in enumerate(pv_power.index):
        hour = timestamp.hour
        
        if hour in peak_hours:
            # Reduce load during grid peak hours
            optimal_load[i] = base_load * 0.7
        elif 10 <= hour < 16:
            # Increase load during high PV production
            optimal_load[i] = base_load * 1.3
        else:
            optimal_load[i] = base_load
    
    # Add some randomness for realism
    optimal_load += np.random.normal(0, 2, len(optimal_load))
    
    return np.clip(optimal_load, 5, base_load * 1.5)

# === SYSTEM OPTIMIZATION CALCULATIONS ===
print("Calculating optimal system sizing and energy management...")

# Calculate current system capabilities
total_panels = sum(seg["num_modules"] for seg in field_segments)
total_pv_capacity_kw = total_panels * panel_power_max / 1000
annual_pv_energy = df["AC_Power_kW_osm_total"].sum() * (5/60)

print(f"Current PV System: {total_panels} panels, {total_pv_capacity_kw:.1f} kW capacity")
print(f"Annual PV Production: {annual_pv_energy:,.0f} kWh")

# Calculate optimal loads
battery_capacity_kwh = 100
battery_power_kw = 15

max_load_no_import, load_with_export = calculate_optimal_load_profile(
    df["AC_Power_kW_osm_total"], battery_capacity_kwh, battery_power_kw
)

# Create optimal load profiles
optimal_load_shifting = create_optimal_load_shifting(df["AC_Power_kW_osm_total"])

# Add to dataframe
df['Max_Load_No_Import_kW'] = max_load_no_import
df['Load_With_Export_kW'] = load_with_export
df['Optimal_Load_Shifting_kW'] = optimal_load_shifting

# Calculate key metrics for each scenario
scenario_metrics = {}

# Scenario 1: Maximum load without imports
total_load_no_import = df['Max_Load_No_Import_kW'].sum() * (5/60)
scenario_metrics['No Import'] = {
    'total_load': total_load_no_import,
    'avg_load': df['Max_Load_No_Import_kW'].mean(),
    'max_load': df['Max_Load_No_Import_kW'].max()
}

# Scenario 2: Load with export capability
total_load_with_export = df['Load_With_Export_kW'].sum() * (5/60)
export_energy = (df[df.index.hour.isin([18, 19, 20])]["AC_Power_kW_osm_total"] - 
                df[df.index.hour.isin([18, 19, 20])]['Load_With_Export_kW']).clip(lower=0).sum() * (5/60)
scenario_metrics['With Export'] = {
    'total_load': total_load_with_export,
    'avg_load': df['Load_With_Export_kW'].mean(),
    'max_load': df['Load_With_Export_kW'].max(),
    'export_energy': export_energy
}

# Scenario 3: Optimal load shifting
total_load_optimal = df['Optimal_Load_Shifting_kW'].sum() * (5/60)
scenario_metrics['Optimal Shifting'] = {
    'total_load': total_load_optimal,
    'avg_load': df['Optimal_Load_Shifting_kW'].mean(),
    'max_load': df['Optimal_Load_Shifting_kW'].max()
}

# === OPTIMAL SYSTEM SIZING RECOMMENDATIONS ===
print("\n" + "="*70)
print("OPTIMAL SYSTEM SIZING AND ENERGY MANAGEMENT ANALYSIS")
print("="*70)

# Calculate required battery capacity based on night load
night_hours = [19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5]
night_load = df[df.index.hour.isin(night_hours)]['Optimal_Load_Shifting_kW'].mean()
required_battery_night = night_load * 10  # 10 hours of autonomy
required_battery_peak = df['Optimal_Load_Shifting_kW'].max() * 4  # 4 hours of peak shaving

optimal_battery = max(required_battery_night, required_battery_peak, 50)  # Minimum 50 kWh

# Calculate optimal inverter size
peak_pv_power = df["AC_Power_kW_osm_total"].max()
peak_load = df['Optimal_Load_Shifting_kW'].max()
optimal_inverter = max(peak_pv_power, peak_load, 30)  # Minimum 30 kW

# Calculate battery amp-hours (assuming 48V system)
battery_voltage = 48
optimal_ah = (optimal_battery * 1000) / battery_voltage

print(f"\nRECOMMENDED OPTIMAL SYSTEM SIZING:")
print(f"Optimal Battery Capacity: {optimal_battery:.0f} kWh")
print(f"Optimal Inverter Size: {optimal_inverter:.0f} kW")
print(f"Battery Amp-Hours ({battery_voltage}V system): {optimal_ah:.0f} Ah")
print(f"PV System Size: {total_pv_capacity_kw:.1f} kW (current)")

print(f"\nSCENARIO ANALYSIS:")
for scenario, metrics in scenario_metrics.items():
    print(f"\n{scenario} Scenario:")
    print(f"  Total Annual Load: {metrics['total_load']:,.0f} kWh")
    print(f"  Average Load: {metrics['avg_load']:.1f} kW")
    print(f"  Peak Load: {metrics['max_load']:.1f} kW")
    if 'export_energy' in metrics:
        print(f"  Evening Export (6-9pm): {metrics['export_energy']:,.0f} kWh")

# === YEARLY PLOT FUNCTIONS ===
def create_yearly_plots(data, filename_prefix='yearly_analysis'):
    """Create yearly plots with month labels on x-axis."""
    
    # Define color scheme
    colors = {
        'pv': '#FF6B00',
        'load_no_import': '#C00000', 
        'load_export': '#38761D',
        'load_optimal': '#1F4E79',
        'export': '#32CD32',
        'import': '#990000',
        'battery': '#FFA500'
    }
    
    # Select only numeric columns for resampling to avoid string columns
    numeric_columns = [
        "AC_Power_kW_osm_total", 
        "Max_Load_No_Import_kW", 
        "Load_With_Export_kW", 
        "Optimal_Load_Shifting_kW"
    ]
    
    # Filter to ensure we only use columns that exist in the dataframe
    available_columns = [col for col in numeric_columns if col in data.columns]
    
    if not available_columns:
        print("No numeric columns found for plotting")
        return
    
    # Create a subset with only numeric columns for resampling
    numeric_data = data[available_columns]
    
    # Resample data to daily and monthly averages for clearer yearly visualization
    daily_data = numeric_data.resample('D').mean()
    monthly_data = numeric_data.resample('M').mean()
    
    # Plot 1: Monthly Average Power Flows
    fig1, ax1 = plt.subplots(figsize=(16, 8))
    
    # Plot monthly averages
    months = monthly_data.index.strftime('%b')
    x_pos = np.arange(len(months))
    
    # Plot available columns
    if "AC_Power_kW_osm_total" in available_columns:
        ax1.plot(x_pos, monthly_data["AC_Power_kW_osm_total"], 
                 label="PV Power", color=colors['pv'], linewidth=3, marker='o', markersize=8)
    
    if "Max_Load_No_Import_kW" in available_columns:
        ax1.plot(x_pos, monthly_data['Max_Load_No_Import_kW'], 
                 label="Max Load (No Import)", color=colors['load_no_import'], linewidth=2, marker='s', markersize=6, linestyle="--")
    
    if "Optimal_Load_Shifting_kW" in available_columns:
        ax1.plot(x_pos, monthly_data['Optimal_Load_Shifting_kW'], 
                 label="Optimal Load Shifting", color=colors['load_optimal'], linewidth=2, marker='^', markersize=6)
    
    ax1.set_ylabel("Average Power [kW]", fontname='Garamond', fontsize=18, fontweight='bold')
    ax1.set_xlabel("Month", fontname='Garamond', fontsize=18, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(months)
    
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center', 
            fontname='Garamond', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_monthly_power.pdf', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='pdf')
    plt.show()
    
    # Plot 2: Daily Load Profiles - Show first 30 days for clarity
    fig2, ax2 = plt.subplots(figsize=(16, 8))
    
    # Take first 30 days for clarity
    first_30_days = numeric_data[(numeric_data.index >= '2024-01-01') & (numeric_data.index < '2024-02-01')]
    
    if "Max_Load_No_Import_kW" in available_columns:
        ax2.plot(first_30_days.index, first_30_days['Max_Load_No_Import_kW'], 
                 label="No Import Load", color=colors['load_no_import'], linewidth=1.5, alpha=0.7)
    
    if "Optimal_Load_Shifting_kW" in available_columns:
        ax2.plot(first_30_days.index, first_30_days['Optimal_Load_Shifting_kW'], 
                 label="Optimal Load", color=colors['load_optimal'], linewidth=1.5)
    
    ax2.set_ylabel("Load Power [kW]", fontname='Garamond', fontsize=18, fontweight='bold')
    ax2.set_xlabel("January 2024", fontname='Garamond', fontsize=18, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(fontsize=14)
    
    # Format x-axis to show days
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
    ax2.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center', 
            fontname='Garamond', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_daily_loads_january.pdf', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='pdf')
    plt.show()
    
    # Plot 3: Monthly Energy Production and Consumption
    fig3, ax3 = plt.subplots(figsize=(16, 8))

    # Calculate monthly energy (sum instead of mean for energy)
    monthly_pv_energy = numeric_data["AC_Power_kW_osm_total"].resample('M').sum() * (5/60) if "AC_Power_kW_osm_total" in available_columns else pd.Series()
    monthly_load_energy = numeric_data['Optimal_Load_Shifting_kW'].resample('M').sum() * (5/60) if "Optimal_Load_Shifting_kW" in available_columns else pd.Series()

    width = 0.35
    x_pos = np.arange(len(months))

    if not monthly_pv_energy.empty and not monthly_load_energy.empty:
        bars1 = ax3.bar(x_pos - width/2, monthly_pv_energy, width, 
                        label='PV Generation', color=colors['pv'], alpha=0.8)
        bars2 = ax3.bar(x_pos + width/2, monthly_load_energy, width, 
                        label='Load Consumption', color=colors['load_optimal'], alpha=0.8)
        
        ax3.set_ylabel("Monthly Energy [kWh]", fontname='Garamond', fontsize=18, fontweight='bold')
        ax3.set_xlabel("Month", fontname='Garamond', fontsize=18, fontweight='bold')
        ax3.grid(True, linestyle='--', alpha=0.3, axis='y')
        ax3.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(months)
        
        # Calculate dynamic positioning for value labels
        max_pv_energy = monthly_pv_energy.max()
        max_load_energy = monthly_load_energy.max()
        y_axis_max = max(max_pv_energy, max_load_energy) * 1.15  # Add 15% margin
        
        # Set y-axis limit to accommodate labels
        ax3.set_ylim(0, y_axis_max)
        
        # Add value labels on bars with dynamic positioning
        def add_value_labels(bars, offset_ratio=0.02):
            for bar in bars:
                height = bar.get_height()
                # Position label slightly above bar top, relative to y-axis max
                label_y = height + (y_axis_max * offset_ratio)
                ax3.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{height:,.0f}', ha='center', va='bottom', 
                    fontname='Garamond', fontsize=10, fontweight='bold',  # Reduced from 12 to 10
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='white',  # Reduced pad from 0.3 to 0.1
                         edgecolor='gray', alpha=0.8, linewidth=0.5))  # Reduced linewidth
        
        add_value_labels(bars1, 0.02)  # PV generation labels
        add_value_labels(bars2, 0.02)  # Load consumption labels

    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0, ha='center', 
            fontname='Garamond', fontsize=14)

    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_monthly_energy.pdf', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='pdf')
    plt.show()
    
    # Plot 4: Seasonal Performance Ratio
    fig4, ax4 = plt.subplots(figsize=(16, 8))
    
    if "AC_Power_kW_osm_total" in available_columns and "Optimal_Load_Shifting_kW" in available_columns:
        # Calculate self-consumption ratio by month
        monthly_export = (numeric_data["AC_Power_kW_osm_total"] - numeric_data['Optimal_Load_Shifting_kW']).clip(lower=0).resample('M').sum() * (5/60)
        monthly_pv_energy = numeric_data["AC_Power_kW_osm_total"].resample('M').sum() * (5/60)
        monthly_self_consumption_ratio = (monthly_pv_energy - monthly_export) / monthly_pv_energy
        
        bars = ax4.bar(x_pos, monthly_self_consumption_ratio * 100, 
                       color=colors['load_optimal'], alpha=0.8)
        
        ax4.set_ylabel("Self-Consumption [%]", fontname='Garamond', fontsize=18, fontweight='bold')
        ax4.set_xlabel("Month", fontname='Garamond', fontsize=18, fontweight='bold')
        ax4.grid(True, linestyle='--', alpha=0.3, axis='y')
        ax4.set_ylim(0, 100)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(months)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.1f}%', ha='center', va='bottom', 
                    fontname='Garamond', fontsize=11, fontweight='bold')
        
        ax4.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='Target (90%)')
        ax4.legend(fontsize=14)
    
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=0, ha='center', 
            fontname='Garamond', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_self_consumption.pdf', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='pdf')
    plt.show()

# === GENERATE YEARLY PLOTS ===
print("\nGenerating yearly analysis plots...")

# Create yearly plots
create_yearly_plots(df, 'yearly_analysis')
