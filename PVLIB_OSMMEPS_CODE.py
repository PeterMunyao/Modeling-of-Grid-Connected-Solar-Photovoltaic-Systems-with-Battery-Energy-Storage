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


# ADDITIONAL PRINTS

print(f"\nFiles saved:")
print("- strategic_export_jan_3_4_osm.pdf (OSM-MEPS model results)")
print("- strategic_export_jan_3_4_pvlib.pdf (PVLib model results)") 
print("- strategic_export_jan_3_4_comparison.pdf (PV power comparison)")
