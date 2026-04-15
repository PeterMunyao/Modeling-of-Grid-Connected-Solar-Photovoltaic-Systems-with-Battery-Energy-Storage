# === IMPORT LIBRARIES ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import pvlib
from pvlib.irradiance import get_total_irradiance
from pvlib.temperature import sapm_cell
from pvlib.pvsystem import pvwatts_dc

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


# === ENSURE REQUIRED COLUMNS EXIST ===
required_columns = ['dni', 'ghi', 'dhi', 'air_temp', 'albedo', 'zenith', 'azimuth',
                    'cloud_opacity', 'relative_humidity', 'wind_speed_10m']
for col in required_columns:
    if col not in df.columns:
        df[col] = 0

# === PV SYSTEM PARAMETERS ===
latitude = -29.815268
longitude = 30.946439
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

# === SOLAR POSITION ===
solar_position = pvlib.solarposition.get_solarposition(df.index, latitude, longitude)

# === INITIALIZE TOTAL POWER COLUMNS ===
df["AC_Power_kW_osm_total"] = 0
df["AC_Power_kW_pvlib_total"] = 0

print("=== PROCESSING SEGMENTS ===")

# === REALISTIC PV SIMULATION PARAMETERS ===
MIN_POA_IRRADIANCE = 50  # W/m² - minimum for meaningful generation
MIN_SOLAR_ELEVATION = 5  # degrees - minimum solar elevation angle
MAX_AOI = 85  # degrees - maximum AOI for any generation

# === LOOP OVER SEGMENTS ===
for i, seg in enumerate(field_segments):
    tilt = seg["tilt"]
    azimuth = seg["azimuth"]
    num_panels = seg["num_modules"]
    
    print(f"Segment {i+1}: {num_panels} panels, tilt={tilt}°, azimuth={azimuth}°")

    # --- PVLIB MODEL ---
    poa = get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        dni=df["dni"],
        ghi=df["ghi"],
        dhi=df["dhi"],
        solar_zenith=solar_position["apparent_zenith"],
        solar_azimuth=solar_position["azimuth"]
    )
    poa_irradiance = poa["poa_global"]
    temp_cell = sapm_cell(poa_irradiance, df["air_temp"], df["wind_speed_10m"], -2.98, -0.0471, 1)
    
    # UNIFORM SCALING: Apply panel count at the DC power level
    dc_power_pvlib = poa_irradiance / stc_irradiance * panel_power_max * num_panels * (1 + temp_coeff * (temp_cell - 25))
    ac_power_pvlib = dc_power_pvlib * inverter_efficiency
    df["AC_Power_kW_pvlib_total"] += ac_power_pvlib / 1000  # Convert to kW

    # --- OSM-MEPS MODEL ---
    tilt_rad = np.radians(tilt)
    az_rad = np.radians(azimuth)
    zen_rad = np.radians(df['zenith'])
    sun_az_rad = np.radians(df['azimuth'])
    
    # Calculate solar elevation
    solar_elevation = 90 - df['zenith']
    
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

    # === REALISTIC GENERATION CUTOFFS ===
    realistic_conditions = (
        (poa_total >= MIN_POA_IRRADIANCE) & 
        (solar_elevation >= MIN_SOLAR_ELEVATION) & 
        (aoi <= MAX_AOI)
    )
    
    # Equation 6: Module temperature model
    module_temp = 45 + poa_total/1000 * (28 - df['air_temp'])
    
    # Equation 7: DC power with temperature correction
    dc_power_osm = panel_power_max * num_panels * (1 + temp_coeff*(module_temp - 45))
    dc_power_osm *= poa_total / stc_irradiance
    
    # Equation 8: Humidity derating
    dc_power_osm *= (1 - 0.002 * df['relative_humidity'])

    # Equation 9: AC power conversion
    ac_power_osm = dc_power_osm * inverter_efficiency
    
    # Apply realistic cutoffs - zero power when conditions aren't met
    ac_power_osm = ac_power_osm.where(realistic_conditions, 0)
    
    # Equation 10: Scale by number of panels
    df["AC_Power_kW_osm_total"] += ac_power_osm / 1000  # Convert to kW

print("PV model calculations completed!")

# Compare the two models
print(f"\nPV Model Comparison:")
print(f"OSM-MEPS Annual Energy: {df['AC_Power_kW_osm_total'].sum() * (5/60):.0f} kWh")
print(f"PVLib Annual Energy: {df['AC_Power_kW_pvlib_total'].sum() * (5/60):.0f} kWh")
difference = abs(df['AC_Power_kW_osm_total'].sum() - df['AC_Power_kW_pvlib_total'].sum()) / df['AC_Power_kW_osm_total'].sum() * 100
print(f"Difference: {difference:.1f}%")

# === BATTERY SYSTEM EQUATIONS ===
battery_capacity_kwh = 150
battery_max_charge_kw = 15
battery_max_discharge_kw = 15
battery_efficiency = 0.90
dt = 5/60

soc_min = 0.10 * battery_capacity_kwh
soc_max = 0.95 * battery_capacity_kwh

# === UPDATED LOAD PROFILE ===
def create_realistic_load(index, base=8, peak=36):
    load = np.ones(len(index)) * base
    
    for i, dt in enumerate(index):
        h = dt.hour
        if 7 <= h < 19:
            load[i] = base + 0.55*(peak - base)
        else:
            load[i] = base * 0.25
    
    load += np.random.normal(0, 1.0, len(load))
    return np.clip(load, 10, peak)

df['Home_Load_kW'] = create_realistic_load(df.index)

# === GRID EXPORT STRATEGY PARAMETERS ===
enable_strategic_export = True
peak_export_hours = [14, 15, 16, 17, 18, 19, 20, 21]
min_soc_for_export = 0.30
max_grid_export_from_battery = 10
export_net_power_threshold = 10

# === STRATEGIC EXPORT SIMULATION FOR BOTH MODELS ===
def run_strategic_export_simulation(pv_power_column):
    """Run battery and grid simulation for a given PV power model"""
    grid_power, battery_power_list, battery_soc_list = [], [], []
    battery_operation_mode = []
    
    # Reset battery SOC for each simulation
    battery_soc = 0.5 * battery_capacity_kwh
    
    for i, (timestamp, row) in enumerate(df.iterrows()):
        pv_power = row[pv_power_column]
        load = row['Home_Load_kW']
        current_hour = timestamp.hour
        
        # Handle NaN values by setting to 0
        if np.isnan(pv_power):
            pv_power = 0
        if np.isnan(load):
            load = 0
        
        net_power = pv_power - load
        battery_power = 0
        operation_mode = "Idle"

        # === MODE 1: BATTERY CHARGING FROM EXCESS PV ===
        if net_power > 0 and battery_soc < soc_max:
            operation_mode = "Charging from PV"
            
            charge_possible = min(net_power, battery_max_charge_kw)
            energy_possible = (soc_max - battery_soc) / battery_efficiency
            charge_power_possible = energy_possible / dt
            charge_power = min(charge_possible, charge_power_possible)
            
            battery_soc += charge_power * dt * battery_efficiency
            net_power -= charge_power
            battery_power = charge_power

        # === MODE 2: BATTERY DISCHARGING TO REDUCE GRID IMPORTS ===
        elif net_power < 0 and battery_soc > soc_min:
            operation_mode = "Discharging to Load"
            
            discharge_needed = min(-net_power, battery_max_discharge_kw)
            energy_available = (battery_soc - soc_min) * battery_efficiency
            discharge_power_possible = energy_available / dt
            discharge_power = min(discharge_needed, discharge_power_possible)
            
            battery_soc -= discharge_power * dt / battery_efficiency
            net_power += discharge_power
            battery_power = -discharge_power

        # === MODE 3: STRATEGIC BATTERY DISCHARGE TO GRID ===
        elif (enable_strategic_export and 
              battery_soc > min_soc_for_export * battery_capacity_kwh and
              current_hour in peak_export_hours and
              battery_soc > soc_min and
              net_power <= export_net_power_threshold):
            
            operation_mode = "Discharging to Grid"
            
            energy_available = (battery_soc - soc_min) * battery_efficiency
            discharge_power_possible = energy_available / dt
            
            if current_hour in [16, 17, 18]:
                target_export = min(max_grid_export_from_battery * 1.2, discharge_power_possible)
            else:
                target_export = min(max_grid_export_from_battery * 0.8, discharge_power_possible)
            
            target_export = min(target_export, battery_max_discharge_kw)
            discharge_power = target_export
            
            battery_soc -= discharge_power * dt / battery_efficiency
            net_power = discharge_power + max(net_power, 0)
            battery_power = -discharge_power

        grid_power.append(net_power)
        battery_power_list.append(battery_power)
        battery_soc_list.append(battery_soc)
        battery_operation_mode.append(operation_mode)
    
    return grid_power, battery_power_list, battery_soc_list, battery_operation_mode

# Run simulations for both models
print("Running strategic export simulation for OSM-MEPS model...")
grid_power_osm, battery_power_osm, battery_soc_osm, battery_mode_osm = run_strategic_export_simulation('AC_Power_kW_osm_total')

print("Running strategic export simulation for PVLib model...")
grid_power_pvlib, battery_power_pvlib, battery_soc_pvlib, battery_mode_pvlib = run_strategic_export_simulation('AC_Power_kW_pvlib_total')

# Add results to DataFrame
df['Grid_Power_KW_osm'] = grid_power_osm
df['Battery_Power_kW_osm'] = battery_power_osm
df['Battery_SOC_kWh_osm'] = battery_soc_osm
df['Battery_Mode_osm'] = battery_mode_osm

df['Grid_Power_KW_pvlib'] = grid_power_pvlib
df['Battery_Power_kW_pvlib'] = battery_power_pvlib
df['Battery_SOC_kWh_pvlib'] = battery_soc_pvlib
df['Battery_Mode_pvlib'] = battery_mode_pvlib

# === ENERGY CALCULATIONS FOR BOTH MODELS ===
def calculate_energy_metrics(pv_column, grid_column, battery_column):
    total_pv_energy = df[pv_column].sum() * dt
    total_load_energy = df['Home_Load_kW'].sum() * dt
    grid_export_energy = df[df[grid_column] > 0][grid_column].sum() * dt
    grid_import_energy = abs(df[df[grid_column] < 0][grid_column].sum() * dt)
    battery_charge_energy = df[df[battery_column] > 0][battery_column].sum() * dt
    battery_discharge_energy = abs(df[df[battery_column] < 0][battery_column].sum() * dt)
    self_consumption_ratio = (total_load_energy - grid_import_energy) / total_load_energy
    
    return {
        'total_pv_energy': total_pv_energy,
        'total_load_energy': total_load_energy,
        'grid_export_energy': grid_export_energy,
        'grid_import_energy': grid_import_energy,
        'battery_charge_energy': battery_charge_energy,
        'battery_discharge_energy': battery_discharge_energy,
        'self_consumption_ratio': self_consumption_ratio
    }

metrics_osm = calculate_energy_metrics('AC_Power_kW_osm_total', 'Grid_Power_KW_osm', 'Battery_Power_kW_osm')
metrics_pvlib = calculate_energy_metrics('AC_Power_kW_pvlib_total', 'Grid_Power_KW_pvlib', 'Battery_Power_kW_pvlib')

# === STRATEGIC EXPORT COMPARISON PLOTS ===
def create_strategic_export_comparison_plots(data, filename_prefix):
    """Create comparison plots for strategic export with both models"""
    
    # Define colors
    colors = {
        'osm_pv': 'orange', 'pvlib_pv': 'maroon', 'load': 'red',
        'battery_charge': '#0047AB', 'battery_discharge': '#1E90FF', 
        'export': '#38761D', 'import': '#990000', 'soc': '#1F4E79',
        'export_fill': '#90EE90', 'import_fill': '#FFB6C1',
    }
    
    # Plot 1: OSM-MEPS Model
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig1.subplots_adjust(hspace=0.25)
    
    # Top plot - Power flows for OSM-MEPS
    power_ylim = [-25, 100]
    
    ax1.plot(data.index, data["AC_Power_kW_osm_total"], 
             label="PV Power (OSM-MEPS)", color=colors['osm_pv'], linewidth=1.5)
    ax1.plot(data.index, data["Home_Load_kW"], 
             label="Load Power", color=colors['load'], linewidth=1.2)
    
    # Battery power
    battery_positive = data['Battery_Power_kW_osm'].clip(lower=0)
    battery_negative = data['Battery_Power_kW_osm'].clip(upper=0)
    ax1.plot(data.index, battery_positive, 
             label="Battery Charging", color=colors['battery_charge'], linewidth=1.2)
    ax1.plot(data.index, battery_negative, 
             label="Battery Discharging", color=colors['battery_discharge'], linewidth=1.2)
    
    # Grid power with area fills
    export_power = data['Grid_Power_KW_osm'].clip(lower=0)
    import_power = data['Grid_Power_KW_osm'].clip(upper=0).abs()
    ax1.fill_between(data.index, export_power, alpha=0.4, color=colors['export_fill'])
    ax1.fill_between(data.index, -import_power, alpha=0.4, color=colors['import_fill'])
    ax1.plot(data.index, export_power, label="Export", color=colors['export'], linewidth=1.0, linestyle='--')
    ax1.plot(data.index, -import_power, label="Import", color=colors['import'], linewidth=1.0, linestyle='--')
    
    ax1.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
    ax1.set_ylim(power_ylim)
    ax1.set_ylabel("Power [kW]", fontname='Garamond', fontsize=16, fontweight='bold')
    ax1.grid(True, which='major', linestyle='-', alpha=0.3, linewidth=0.8)
    ax1.grid(True, which='minor', linestyle='--', alpha=0.2, linewidth=0.5)
    
    # X-axis formatting
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    ax1.xaxis.set_minor_locator(mdates.HourLocator(interval=2))
    
    def custom_date_formatter(x, pos=None):
        dt = mdates.num2date(x)
        if dt.hour == 0:
            return f"Jan/{dt.day:02d}\n00:00"
        else:
            return f"{dt.hour:02d}:00"
    
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(custom_date_formatter))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center', fontname='Garamond', fontsize=12)
    
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), 
               ncol=4, framealpha=0.9, fontsize=10, fancybox=True, shadow=True)
    ax1.set_title("Strategic Export - OSM-MEPS Model", fontname='Garamond', fontsize=18, fontweight='bold')
    
    # Bottom plot - SOC for OSM-MEPS
    ax2.plot(data.index, data["Battery_SOC_kWh_osm"], 
             label="Battery SoC", color=colors['soc'], linewidth=1.5)
    ax2.axhline(soc_max, color='red', linestyle='--', linewidth=1.2, alpha=0.8, label='Max SoC')
    ax2.axhline(soc_min, color='red', linestyle='--', linewidth=1.2, alpha=0.8, label='Min SoC')
    
    ax2.set_ylim([0, battery_capacity_kwh * 1.05])
    ax2.set_ylabel("SoC [kWh]", fontname='Garamond', fontsize=16, fontweight='bold')
    ax2.set_xlabel("Time", fontname='Garamond', fontsize=16, fontweight='bold')
    ax2.grid(True, which='major', linestyle='-', alpha=0.3, linewidth=0.8)
    ax2.grid(True, which='minor', linestyle='--', alpha=0.2, linewidth=0.5)
    
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    ax2.xaxis.set_minor_locator(mdates.HourLocator(interval=2))
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(custom_date_formatter))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center', fontname='Garamond', fontsize=12)
    
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), 
               ncol=3, framealpha=0.9, fontsize=10, fancybox=True, shadow=True)
    
    xlim = (data.index.min(), data.index.max())
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_osm.pdf', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='pdf')
    plt.show()
    
    # Plot 2: PVLib Model
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig2.subplots_adjust(hspace=0.25)
    
    # Top plot - Power flows for PVLib
    ax1.plot(data.index, data["AC_Power_kW_pvlib_total"], 
             label="PV Power (PVLib)", color=colors['pvlib_pv'], linewidth=1.5)
    ax1.plot(data.index, data["Home_Load_kW"], 
             label="Load Power", color=colors['load'], linewidth=1.2)
    
    # Battery power
    battery_positive = data['Battery_Power_kW_pvlib'].clip(lower=0)
    battery_negative = data['Battery_Power_kW_pvlib'].clip(upper=0)
    ax1.plot(data.index, battery_positive, 
             label="Battery Charging", color=colors['battery_charge'], linewidth=1.2)
    ax1.plot(data.index, battery_negative, 
             label="Battery Discharging", color=colors['battery_discharge'], linewidth=1.2)
    
    # Grid power with area fills
    export_power = data['Grid_Power_KW_pvlib'].clip(lower=0)
    import_power = data['Grid_Power_KW_pvlib'].clip(upper=0).abs()
    ax1.fill_between(data.index, export_power, alpha=0.4, color=colors['export_fill'])
    ax1.fill_between(data.index, -import_power, alpha=0.4, color=colors['import_fill'])
    ax1.plot(data.index, export_power, label="Export", color=colors['export'], linewidth=1.0, linestyle='--')
    ax1.plot(data.index, -import_power, label="Import", color=colors['import'], linewidth=1.0, linestyle='--')
    
    ax1.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
    ax1.set_ylim(power_ylim)
    ax1.set_ylabel("Power [kW]", fontname='Garamond', fontsize=16, fontweight='bold')
    ax1.grid(True, which='major', linestyle='-', alpha=0.3, linewidth=0.8)
    ax1.grid(True, which='minor', linestyle='--', alpha=0.2, linewidth=0.5)
    
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    ax1.xaxis.set_minor_locator(mdates.HourLocator(interval=2))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(custom_date_formatter))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center', fontname='Garamond', fontsize=12)
    
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), 
               ncol=4, framealpha=0.9, fontsize=10, fancybox=True, shadow=True)
    ax1.set_title("Strategic Export - PVLib Model", fontname='Garamond', fontsize=18, fontweight='bold')
    
    # Bottom plot - SOC for PVLib
    ax2.plot(data.index, data["Battery_SOC_kWh_pvlib"], 
             label="Battery SoC", color=colors['soc'], linewidth=1.5)
    ax2.axhline(soc_max, color='red', linestyle='--', linewidth=1.2, alpha=0.8, label='Max SoC')
    ax2.axhline(soc_min, color='red', linestyle='--', linewidth=1.2, alpha=0.8, label='Min SoC')
    
    ax2.set_ylim([0, battery_capacity_kwh * 1.05])
    ax2.set_ylabel("SoC [kWh]", fontname='Garamond', fontsize=16, fontweight='bold')
    ax2.set_xlabel("Time", fontname='Garamond', fontsize=16, fontweight='bold')
    ax2.grid(True, which='major', linestyle='-', alpha=0.3, linewidth=0.8)
    ax2.grid(True, which='minor', linestyle='--', alpha=0.2, linewidth=0.5)
    
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    ax2.xaxis.set_minor_locator(mdates.HourLocator(interval=2))
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(custom_date_formatter))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center', fontname='Garamond', fontsize=12)
    
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), 
               ncol=3, framealpha=0.9, fontsize=10, fancybox=True, shadow=True)
    
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_pvlib.pdf', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='pdf')
    plt.show()
    
    # Plot 3: Combined PV Power Comparison
    fig3, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot(data.index, data["AC_Power_kW_osm_total"], 
            label="PV Power (OSM-MEPS)", color=colors['osm_pv'], linewidth=2.2)
    ax.plot(data.index, data["AC_Power_kW_pvlib_total"], 
            label="PV Power (PVLIB)", color=colors['pvlib_pv'], linewidth=2.2, linestyle='--')
    ax.plot(data.index, data["Home_Load_kW"], 
            label="Load Power", color=colors['load'], linewidth=1.8, alpha=0.7)
    
    ax.set_ylabel("Power [kW]", fontname='Garamond', fontsize=16, fontweight='bold')
    ax.set_xlabel("Time", fontname='Garamond', fontsize=16, fontweight='bold')
    ax.grid(True, which='major', linestyle='-', alpha=0.5, linewidth=0.9)
    ax.grid(True, which='minor', linestyle='--', alpha=0.4, linewidth=0.7)
    
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(custom_date_formatter))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center', fontname='Garamond', fontsize=12)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
              ncol=3, framealpha=0.9, fontsize=12, fancybox=True, shadow=True)
    ax.set_xlim(xlim)
    
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_comparison.pdf', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='pdf')
    plt.show()

# Generate strategic export comparison plots for January 3-4
jan_3_4_data = df[(df.index >= '2024-01-03') & (df.index < '2024-01-05')]
print("Generating strategic export comparison plots...")
create_strategic_export_comparison_plots(jan_3_4_data, 'strategic_export_jan_3_4')

# === SYSTEM STATISTICS COMPARISON ===
print("\n" + "="*70)
print("SYSTEM PERFORMANCE COMPARISON: OSM-MEPS vs PVLib")
print("="*70)

print(f"{'Metric':<25} {'OSM-MEPS':<12} {'PVLib':<12} {'Difference':<12}")
print("-" * 70)
print(f"{'PV Energy (kWh)':<25} {metrics_osm['total_pv_energy']:>11.0f} {metrics_pvlib['total_pv_energy']:>11.0f} {abs(metrics_osm['total_pv_energy'] - metrics_pvlib['total_pv_energy']):>11.0f}")
print(f"{'Grid Export (kWh)':<25} {metrics_osm['grid_export_energy']:>11.0f} {metrics_pvlib['grid_export_energy']:>11.0f} {abs(metrics_osm['grid_export_energy'] - metrics_pvlib['grid_export_energy']):>11.0f}")
print(f"{'Grid Import (kWh)':<25} {metrics_osm['grid_import_energy']:>11.0f} {metrics_pvlib['grid_import_energy']:>11.0f} {abs(metrics_osm['grid_import_energy'] - metrics_pvlib['grid_import_energy']):>11.0f}")
print(f"{'Self-Consumption (%)':<25} {metrics_osm['self_consumption_ratio']:>11.1%} {metrics_pvlib['self_consumption_ratio']:>11.1%} {abs(metrics_osm['self_consumption_ratio'] - metrics_pvlib['self_consumption_ratio']):>11.1%}")

print(f"\nFiles saved:")
print("- strategic_export_jan_3_4_osm.pdf (OSM-MEPS model results)")
print("- strategic_export_jan_3_4_pvlib.pdf (PVLib model results)") 
print("- strategic_export_jan_3_4_comparison.pdf (PV power comparison)")

#-----------------------------------------------------------------------------------------------------------------------------------


# === CREATE IEEE COLUMN-SIZED YEARLY PLOT WITH ORIGINAL STYLE ===
def create_ieee_column_plot_original_style(data, battery_capacity_kwh, soc_max, power_max, filename='yearly_ieee_column'):
    """Create IEEE column-sized plot with original style but only PV, Export, Import power and SoC"""
    
    # Set IEEE column size (3.5 inches wide) but maintain original styling
    fig_width = 8.0  # IEEE single column width in inches
    fig_height = 7.0  # Adjusted height for two subplots
    
    # Create figure with IEEE column dimensions but original styling
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, fig_height))
    fig.subplots_adjust(hspace=0.3)  # Reduced spacing to bring legends closer
    
    # Use original color scheme
    colors = {
        'pv': '#FF6B00',           # Bright orange for PV
        'export': '#38761D',       # Green for export
        'import': '#990000',       # Dark red for import
        'soc': '#1F4E79',          # Navy blue for SOC
        'export_fill': '#90EE90',  # Light green for export area
        'import_fill': '#FFB6C1',  # Light red for import area
    }
    
    # --- TOP PLOT: POWER FLOWS WITH AREA FILLS ---
    
    # Set power y-axis limits to eliminate empty space below -30 kW
    power_ylim = [-30, power_max]  # Limit lower bound to -30 kW
    
    # Plot PV Power (maintain original linewidth and style)
    ax1.plot(data.index, data["AC_Power_kW_osm_total"], 
             label="PV Power", color=colors['pv'], linewidth=1.2)
    
    # Highlight Grid Export and Import with area fills (original style)
    # FIXED: Use correct column name 'Grid_Power_KW_osm'
    export_power = data['Grid_Power_KW_osm'].clip(lower=0)
    import_power = data['Grid_Power_KW_osm'].clip(upper=0).abs()
    
    # Area fills for export and import (original alpha values)
    ax1.fill_between(data.index, export_power, alpha=0.4, 
                     color=colors['export_fill'], label='Grid Export')
    ax1.fill_between(data.index, -import_power, alpha=0.4, 
                     color=colors['import_fill'], label='Grid Import')
    
    # Line plots on top of area fills (original line styles)
    ax1.plot(data.index, export_power, 
             label="Export Power", color=colors['export'], linewidth=1.0, linestyle='--')
    ax1.plot(data.index, -import_power, 
             label="Import Power", color=colors['import'], linewidth=1.0, linestyle='--')
    
    # Zero line (original style)
    ax1.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
    
    # Set consistent power y-axis limits with -30 kW lower bound
    ax1.set_ylim(power_ylim)
    
    # Maintain original font styling but adjust sizes for IEEE column
    ax1.set_ylabel("Power [kW]", fontname='Garamond', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # X-axis formatting for yearly plot - show all months
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    # Ensure all months are labeled
    months = mdates.MonthLocator()
    ax1.xaxis.set_major_locator(months)
    
    # Adjust x-tick label font size and alignment
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center', 
             fontname='Garamond', fontsize=12)  # Increased font size
    
    # Remove x-axis label for top plot
    ax1.set_xlabel("")
    
    # Combined power legend - moved closer to graph
    power_legend = ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
               ncol=3, framealpha=0.9, fontsize=10, fancybox=True, shadow=True)
    
    # --- BOTTOM PLOT: BATTERY SOC WITH ORIGINAL STYLING ---
    
    # Plot Battery SOC (original linewidth)
    # FIXED: Use correct column name 'Battery_SOC_kWh_osm'
    ax2.plot(data.index, data["Battery_SOC_kWh_osm"], 
             label="Battery SOC", color=colors['soc'], linewidth=1.5)
    
    # SOC limits (original styling)
    ax2.axhline(soc_max, color='red', linestyle='--', linewidth=1.2, alpha=0.8, label='SOC Limits')
    
    # SOC plot formatting (original scale alignment)
    soc_range = battery_capacity_kwh
    soc_ylim = [0, soc_range * 1.05]
    
    ax2.set_ylim(soc_ylim)
    
    # Maintain original font styling but adjust sizes
    ax2.set_ylabel("Battery SOC [kWh]", fontname='Garamond', fontsize=14, fontweight='bold')
    ax2.set_xlabel("")  # REMOVED "Month" label
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # X-axis formatting (same as top plot) - show all months
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    # Ensure all months are labeled
    ax2.xaxis.set_major_locator(months)
    
    # Adjust x-tick label font size and alignment
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center', 
             fontname='Garamond', fontsize=12)
    
    # Combined SOC legend - moved closer to graph
    soc_legend = ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
               ncol=2, framealpha=0.9, fontsize=10, fancybox=True, shadow=True)
    
    # Ensure both plots have the same x-axis limits
    xlim = (data.index.min(), data.index.max())
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    
    plt.tight_layout()
    
    # Save as PDF with high quality (original settings)
    plt.savefig(f'{filename}.pdf', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='pdf')
    
    # Also save as high-quality PNG for review
    plt.savefig(f'{filename}.png', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='png')
    
    return fig


# === CREATE DAILY DETAIL PLOT WITH ORIGINAL STYLE ===
def create_daily_ieee_column_plot(data, battery_capacity_kwh, soc_max, filename='daily_ieee_column'):
    """Create IEEE column-sized daily plot with original style"""
    
    # Set IEEE column size
    fig_width = 5.5
    fig_height = 7.0
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, fig_height))
    fig.subplots_adjust(hspace=0.3)  # Reduced spacing
    
    # Use original color scheme
    colors = {
        'pv': '#FF6B00',           # Bright orange for PV
        'export': '#38761D',       # Green for export
        'import': '#990000',       # Dark red for import
        'soc': '#1F4E79',          # Navy blue for SOC
        'export_fill': '#90EE90',  # Light green for export area
        'import_fill': '#FFB6C1',  # Light red for import area
    }
    
    # --- TOP PLOT: POWER FLOWS ---
    
    # Plot PV Power
    ax1.plot(data.index, data["AC_Power_kW_osm_total"], 
             label="PV Power", color=colors['pv'], linewidth=1.2)
    
    # Highlight Grid Export and Import with area fills
    # FIXED: Use correct column name 'Grid_Power_KW_osm'
    export_power = data['Grid_Power_KW_osm'].clip(lower=0)
    import_power = data['Grid_Power_KW_osm'].clip(upper=0).abs()
    
    ax1.fill_between(data.index, export_power, alpha=0.4, 
                     color=colors['export_fill'], label='Grid Export')
    ax1.fill_between(data.index, -import_power, alpha=0.4, 
                     color=colors['import_fill'], label='Grid Import')
    
    # Line plots on top of area fills
    ax1.plot(data.index, export_power, 
             label="Export Power", color=colors['export'], linewidth=1.0, linestyle='--')
    ax1.plot(data.index, -import_power, 
             label="Import Power", color=colors['import'], linewidth=1.0, linestyle='--')
    
    ax1.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
    
    # Calculate power limits for this specific day with -30 kW lower bound
    day_power_max = max(data["AC_Power_kW_osm_total"].max(), 
                       export_power.max(), 
                       import_power.max()) * 1.1
    ax1.set_ylim([-30, day_power_max])  # Limit lower bound to -30 kW
    
    ax1.set_ylabel("Power [kW]", fontname='Garamond', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # X-axis formatting for daily plot - show all hours clearly
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center', 
             fontname='Garamond', fontsize=10)
    
    # Remove x-axis label for top plot
    ax1.set_xlabel("")
    
    # Combined power legend - moved closer
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
               ncol=3, framealpha=0.9, fontsize=9, fancybox=True, shadow=True)
    
    # --- BOTTOM PLOT: BATTERY SOC ---
    
    # FIXED: Use correct column name 'Battery_SOC_kWh_osm'
    ax2.plot(data.index, data["Battery_SOC_kWh_osm"], 
             label="Battery SOC", color=colors['soc'], linewidth=1.5)
    
    ax2.axhline(soc_max, color='red', linestyle='--', linewidth=1.2, alpha=0.8, label='SOC Limits')
    
    ax2.set_ylim([0, battery_capacity_kwh * 1.05])
    ax2.set_ylabel("Battery SOC [kWh]", fontname='Garamond', fontsize=12, fontweight='bold')
    ax2.set_xlabel("Time of Day", fontname='Garamond', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # X-axis formatting - show all hours clearly
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center',
             fontname='Garamond', fontsize=10)
    
    # Combined SOC legend - moved closer
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
               ncol=2, framealpha=0.9, fontsize=9, fancybox=True, shadow=True)
    
    # Ensure both plots have the same x-axis limits
    xlim = (data.index.min(), data.index.max())
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    
    plt.tight_layout()
    
    # Save plots
    plt.savefig(f'{filename}.pdf', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='pdf')
    plt.savefig(f'{filename}.png', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='png')
    
    return fig


# === GENERATE IEEE COLUMN PLOTS WITH ORIGINAL STYLE ===
print("Generating IEEE column-sized plots with original style...")

# Use yearly data
yearly_data = df[(df.index >= '2024-01-01') & (df.index < '2025-01-01')]

# Calculate power_max for yearly plot
all_power_data = np.concatenate([
    yearly_data["AC_Power_kW_osm_total"].values,
    yearly_data['Grid_Power_KW_osm'].clip(lower=0).values,
    yearly_data['Grid_Power_KW_osm'].clip(upper=0).abs().values
])
power_max = np.max(np.abs(all_power_data)) * 1.1

# Create the IEEE column yearly plot
fig_yearly_ieee = create_ieee_column_plot_original_style(
    yearly_data, battery_capacity_kwh, soc_max, power_max, 'yearly_ieee_column'
)
plt.show()
plt.close(fig_yearly_ieee)

# Create daily detailed plots for key periods (corrected for Southern Hemisphere)
daily_periods = [
    ('2024-06-15', '2024-06-16', 'winter_day'),     # June = Winter in Southern Hemisphere
    ('2024-12-15', '2024-12-16', 'summer_day'),     # December = Summer in Southern Hemisphere
]

for start_date, end_date, period_name in daily_periods:
    daily_data = df[(df.index >= start_date) & (df.index < end_date)]
    if len(daily_data) > 0:
        fig_daily = create_daily_ieee_column_plot(
            daily_data, battery_capacity_kwh, soc_max, f'{period_name}_ieee_column'
        )
        plt.show()
        plt.close(fig_daily)

print("\nIEEE column-sized plots with original style generated successfully!")
print("Files saved:")
print("- yearly_ieee_column.pdf & .png")
print("- winter_day_ieee_column.pdf & .png (June 15-16 - Southern Hemisphere Winter)") 
print("- summer_day_ieee_column.pdf & .png (December 15-16 - Southern Hemisphere Summer)")

#----------------------------------------------------------------------------------------

# === PLOT FOR JANUARY 3RD AND 4TH ONLY ===
jan_3_4_data = df[(df.index >= '2024-01-03') & (df.index < '2024-01-05')]

# Create compact plot for Jan 3-4
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))
fig.subplots_adjust(hspace=0.25)  # Adjusted spacing between subplots

# Define colors
colors = {
    'pv': '#FF6B00', 'load': '#C00000', 'battery_charge': '#0047AB',
    'battery_discharge': '#1E90FF', 'export': '#38761D', 'import': '#990000',
    'soc': '#1F4E79', 'export_fill': '#90EE90', 'import_fill': '#FFB6C1',
}

# TOP PLOT: POWER FLOWS
# Set power y-axis limits with minimum negative power at -25kW
power_ylim = [-25, 100]

# Plot components
ax1.plot(jan_3_4_data.index, jan_3_4_data["AC_Power_kW_osm_total"], 
         label="PV Power", color=colors['pv'], linewidth=1.2)
ax1.plot(jan_3_4_data.index, jan_3_4_data["Home_Load_kW"], 
         label="Load Power", color=colors['load'], linewidth=1.2)

# Battery power - USING CORRECT COLUMN NAME 'Battery_Power_kW_osm'
battery_positive = jan_3_4_data['Battery_Power_kW_osm'].clip(lower=0)
battery_negative = jan_3_4_data['Battery_Power_kW_osm'].clip(upper=0)
ax1.plot(jan_3_4_data.index, battery_positive, 
         label="Battery Charging", color=colors['battery_charge'], linewidth=1.2)
ax1.plot(jan_3_4_data.index, battery_negative, 
         label="Battery Discharging", color=colors['battery_discharge'], linewidth=1.2)

# Grid power with area fills - USING CORRECT COLUMN NAME 'Grid_Power_KW_osm'
export_power = jan_3_4_data['Grid_Power_KW_osm'].clip(lower=0)
import_power = jan_3_4_data['Grid_Power_KW_osm'].clip(upper=0).abs()
ax1.fill_between(jan_3_4_data.index, export_power, alpha=0.4, color=colors['export_fill'])
ax1.fill_between(jan_3_4_data.index, -import_power, alpha=0.4, color=colors['import_fill'])
ax1.plot(jan_3_4_data.index, export_power, label="Export", color=colors['export'], linewidth=1.0, linestyle='--')
ax1.plot(jan_3_4_data.index, -import_power, label="Import", color=colors['import'], linewidth=1.0, linestyle='--')

ax1.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
ax1.set_ylim(power_ylim)
ax1.set_ylabel("Power [kW]", fontname='Garamond', fontsize=16, fontweight='bold')

# Configure grids - major and minor grids every 4 hours
ax1.grid(True, which='major', linestyle='-', alpha=0.3, linewidth=0.8)
ax1.grid(True, which='minor', linestyle='--', alpha=0.2, linewidth=0.5)

# X-axis formatting for 2-day period
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))
ax1.xaxis.set_minor_locator(mdates.HourLocator(interval=2))  # Minor grids every 2 hours

# Custom formatter for dates - simplified approach
def custom_date_formatter(x, pos=None):
    dt = mdates.num2date(x)
    # Show date only at midnight, time at other hours
    if dt.hour == 0:
        return f"Jan/{dt.day:02d}\n00:00"
    else:
        return f"{dt.hour:02d}:00"

ax1.xaxis.set_major_formatter(plt.FuncFormatter(custom_date_formatter))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center', fontname='Garamond', fontsize=12)

# Legend below the top plot - REDUCED ncol to 5 since we have 5 items
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), 
           ncol=5, framealpha=0.9, fontsize=11, fancybox=True, shadow=True,
           columnspacing=0.8, handlelength=1.2)

# BOTTOM PLOT: BATTERY SOC - USING CORRECT COLUMN NAME 'Battery_SOC_kWh_osm'
ax2.plot(jan_3_4_data.index, jan_3_4_data["Battery_SOC_kWh_osm"], 
         label="Battery SoC", color=colors['soc'], linewidth=1.5)
ax2.axhline(soc_max, color='red', linestyle='--', linewidth=1.2, alpha=0.8, label='Max SoC')
ax2.axhline(soc_min, color='red', linestyle='--', linewidth=1.2, alpha=0.8, label='Min SoC')

ax2.set_ylim([0, battery_capacity_kwh * 1.05])
ax2.set_ylabel("SoC [kWh]", fontname='Garamond', fontsize=16, fontweight='bold')
ax2.set_xlabel("Time", fontname='Garamond', fontsize=16, fontweight='bold')

# Configure grids - major and minor grids every 4 hours
ax2.grid(True, which='major', linestyle='-', alpha=0.3, linewidth=0.8)
ax2.grid(True, which='minor', linestyle='--', alpha=0.2, linewidth=0.5)

# X-axis formatting (same as top plot)
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=4))
ax2.xaxis.set_minor_locator(mdates.HourLocator(interval=2))  # Minor grids every 2 hours
ax2.xaxis.set_major_formatter(plt.FuncFormatter(custom_date_formatter))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center', fontname='Garamond', fontsize=12)

# Legend below the bottom plot
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), 
           ncol=3, framealpha=0.9, fontsize=11, fancybox=True, shadow=True,
           columnspacing=1.0, handlelength=1.5)

# Set same x-axis limits for both plots
xlim = (jan_3_4_data.index.min(), jan_3_4_data.index.max())
ax1.set_xlim(xlim)
ax2.set_xlim(xlim)

plt.tight_layout()
plt.savefig('january_3_4_compact.pdf', dpi=600, bbox_inches='tight', 
            facecolor='white', edgecolor='none', format='pdf')
plt.show()

print("January 3-4 compact plot generated successfully!")
