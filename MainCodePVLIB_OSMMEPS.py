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

# Set Garamond bold font for all labels
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
df['period_end'] = pd.to_datetime(df['period_end'])
df.set_index('period_end', inplace=True)
df = df[(df.index >= '2024-01-01') & (df.index < '2025-01-01')]

# NO TIME SHIFT - solar position calculated from actual timestamps
# df.index = df.index + pd.Timedelta(hours=2)  # REMOVED - causes misalignment

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

# === SOLAR POSITION (calculated from actual timestamps) ===
solar_position = pvlib.solarposition.get_solarposition(df.index, latitude, longitude)

# === INITIALIZE TOTAL POWER COLUMNS ===
df["AC_Power_kW_osm_total"] = 0
df["AC_Power_kW_pvlib_total"] = 0

print("=== PROCESSING SEGMENTS ===")

# === REALISTIC PV SIMULATION PARAMETERS ===
MIN_POA_IRRADIANCE = 50      # W/m² - minimum for meaningful generation
MIN_SOLAR_ELEVATION = 5      # degrees - minimum solar elevation angle
MAX_AOI = 85                 # degrees - maximum AOI for any generation

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
    
    # DC power with temperature correction and clipping
    dc_power_pvlib = poa_irradiance / stc_irradiance * panel_power_max * num_panels * (1 + temp_coeff * (temp_cell - 25))
    dc_power_pvlib = np.clip(dc_power_pvlib, 0, panel_power_max * num_panels)
    ac_power_pvlib = dc_power_pvlib * inverter_efficiency
    df["AC_Power_kW_pvlib_total"] += ac_power_pvlib / 1000

    # --- OSM-MEPS MODEL ---
    tilt_rad = np.radians(tilt)
    az_rad = np.radians(azimuth)
    zen_rad = np.radians(df['zenith'])
    sun_az_rad = np.radians(df['azimuth'])
    
    # Solar elevation
    solar_elevation = 90 - df['zenith']
    
    # Angle of Incidence (AOI)
    aoi = np.degrees(np.arccos(
        np.cos(zen_rad) * np.cos(tilt_rad) +
        np.sin(zen_rad) * np.sin(tilt_rad) * np.cos(sun_az_rad - az_rad)
    ))
    aoi = np.clip(aoi, 0, 90)

    # Plane of Array (POA) Irradiance
    poa_direct = df['dni'] * np.cos(np.radians(aoi)) * (1 - df['cloud_opacity']/100)
    poa_direct = poa_direct.clip(lower=0)
    poa_diffuse = df['dhi'] * (1 + np.cos(tilt_rad)) / 2
    poa_reflected = df['ghi'] * df['albedo'] * (1 - np.cos(tilt_rad)) / 2
    poa_total = poa_direct + poa_diffuse + poa_reflected

    # Realistic generation cutoffs
    realistic_conditions = (
        (poa_total >= MIN_POA_IRRADIANCE) & 
        (solar_elevation >= MIN_SOLAR_ELEVATION) & 
        (aoi <= MAX_AOI)
    )
    
    # Module temperature model (corrected reference temperature)
    module_temp = 45 + poa_total/1000 * (28 - df['air_temp'])
    
    # DC power with temperature correction (reference 25°C for STC)
    dc_power_osm = panel_power_max * num_panels * (1 + temp_coeff * (module_temp - 25))
    dc_power_osm *= poa_total / stc_irradiance
    dc_power_osm = np.clip(dc_power_osm, 0, panel_power_max * num_panels)
    
    # Humidity derating
    dc_power_osm *= (1 - 0.002 * df['relative_humidity'])

    # AC power conversion
    ac_power_osm = dc_power_osm * inverter_efficiency
    
    # Apply realistic cutoffs
    ac_power_osm = ac_power_osm.where(realistic_conditions, 0)
    
    # Scale by number of panels
    df["AC_Power_kW_osm_total"] += ac_power_osm / 1000

print("PV model calculations completed!")

# === COMPARE THE TWO MODELS ===
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
dt = 5/60  # 5 minutes in hours

soc_min = 0.10 * battery_capacity_kwh
soc_max = 0.95 * battery_capacity_kwh

# === LOAD PROFILE ===
def create_realistic_load(index, base=8, peak=36):
    load = np.ones(len(index)) * base
    
    for i, dt in enumerate(index):
        h = dt.hour
        if 7 <= h < 19:
            load[i] = base + 0.55 * (peak - base)
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

# === STRATEGIC EXPORT SIMULATION (CORRECTED BATTERY LOGIC) ===
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
        
        # Handle NaN values
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
            # CORRECTED: Energy that can be stored in battery
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
            # CORRECTED: Energy available from battery
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
            
            # CORRECTED: Energy available from battery
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

# Run simulations for OSM-MEPS model
print("Running strategic export simulation for OSM-MEPS model...")
grid_power_osm, battery_power_osm, battery_soc_osm, battery_mode_osm = run_strategic_export_simulation('AC_Power_kW_osm_total')

# Use OSM-MEPS results for all subsequent plots (as primary model)
df['Grid_Power_KW'] = grid_power_osm
df['Battery_Power_kW'] = battery_power_osm
df['Battery_SOC_kWh'] = battery_soc_osm
df['Battery_Mode'] = battery_mode_osm

# === ENERGY CALCULATIONS ===
total_pv_energy = df['AC_Power_kW_osm_total'].sum() * dt
total_load_energy = df['Home_Load_kW'].sum() * dt
grid_export_energy = df[df['Grid_Power_KW'] > 0]['Grid_Power_KW'].sum() * dt
grid_import_energy = abs(df[df['Grid_Power_KW'] < 0]['Grid_Power_KW'].sum() * dt)
battery_charge_energy = df[df['Battery_Power_kW'] > 0]['Battery_Power_kW'].sum() * dt
battery_discharge_energy = abs(df[df['Battery_Power_kW'] < 0]['Battery_Power_kW'].sum() * dt)
self_consumption_ratio = (total_load_energy - grid_import_energy) / total_load_energy

print("\n" + "="*70)
print("SYSTEM PERFORMANCE SUMMARY (OSM-MEPS Model)")
print("="*70)
print(f"{'PV Energy (kWh)':<25} {total_pv_energy:>11.0f}")
print(f"{'Total Load (kWh)':<25} {total_load_energy:>11.0f}")
print(f"{'Grid Export (kWh)':<25} {grid_export_energy:>11.0f}")
print(f"{'Grid Import (kWh)':<25} {grid_import_energy:>11.0f}")
print(f"{'Battery Charge (kWh)':<25} {battery_charge_energy:>11.0f}")
print(f"{'Battery Discharge (kWh)':<25} {battery_discharge_energy:>11.0f}")
print(f"{'Self-Consumption Ratio':<25} {self_consumption_ratio:>11.1%}")
print("="*70)

# === IEEE COLUMN-SIZED PLOTTING FUNCTIONS ===

def create_ieee_column_plot(data, battery_capacity_kwh, soc_max, soc_min, 
                            filename='yearly_ieee_column'):
    """Create IEEE column-sized plot with PV, Export, Import power and SoC"""
    
    # IEEE single column width (3.5 inches) - publication ready
    fig_width = 3.5
    fig_height = 6.0
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, fig_height))
    fig.subplots_adjust(hspace=0.25)
    
    # Color scheme
    colors = {
        'pv': '#FF6B00',           # Bright orange for PV
        'export': '#38761D',       # Green for export
        'import': '#990000',       # Dark red for import
        'soc': '#1F4E79',          # Navy blue for SOC
        'export_fill': '#90EE90',  # Light green for export area
        'import_fill': '#FFB6C1',  # Light red for import area
    }
    
    # Calculate power limits
    all_power_data = np.concatenate([
        data["AC_Power_kW_osm_total"].values,
        data['Grid_Power_KW'].clip(lower=0).values,
        data['Grid_Power_KW'].clip(upper=0).abs().values
    ])
    power_max = np.max(np.abs(all_power_data)) * 1.1
    
    # --- TOP PLOT: POWER FLOWS ---
    power_ylim = [-30, power_max]
    
    # PV Power
    ax1.plot(data.index, data["AC_Power_kW_osm_total"], 
             label="PV Power", color=colors['pv'], linewidth=1.0)
    
    # Grid Export and Import with area fills
    export_power = data['Grid_Power_KW'].clip(lower=0)
    import_power = data['Grid_Power_KW'].clip(upper=0).abs()
    
    ax1.fill_between(data.index, export_power, alpha=0.3, 
                     color=colors['export_fill'], label='Grid Export')
    ax1.fill_between(data.index, -import_power, alpha=0.3, 
                     color=colors['import_fill'], label='Grid Import')
    
    # Line plots
    ax1.plot(data.index, export_power, color=colors['export'], linewidth=0.8, linestyle='--')
    ax1.plot(data.index, -import_power, color=colors['import'], linewidth=0.8, linestyle='--')
    
    ax1.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)
    ax1.set_ylim(power_ylim)
    ax1.set_ylabel("Power [kW]", fontname='Garamond', fontsize=9, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    
    # X-axis formatting
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center', 
             fontname='Garamond', fontsize=8)
    ax1.set_xlabel("")
    
    # Legend
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), 
               ncol=3, framealpha=0.9, fontsize=7, fancybox=True, shadow=True)
    
    # --- BOTTOM PLOT: BATTERY SOC ---
    ax2.plot(data.index, data["Battery_SOC_kWh"], 
             label="Battery SOC", color=colors['soc'], linewidth=1.0)
    ax2.axhline(soc_max, color='red', linestyle='--', linewidth=0.8, alpha=0.8, label='SOC Limits')
    ax2.axhline(soc_min, color='red', linestyle='--', linewidth=0.8, alpha=0.8)
    
    ax2.set_ylim([0, battery_capacity_kwh * 1.05])
    ax2.set_ylabel("Battery SOC [kWh]", fontname='Garamond', fontsize=9, fontweight='bold')
    ax2.set_xlabel("Month", fontname='Garamond', fontsize=9, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    
    # X-axis formatting
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center', 
             fontname='Garamond', fontsize=8)
    
    # Legend
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), 
               ncol=2, framealpha=0.9, fontsize=7, fancybox=True, shadow=True)
    
    # Set x-limits
    xlim = (data.index.min(), data.index.max())
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(f'{filename}.pdf', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='pdf')
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='png')
    
    return fig

def create_daily_ieee_column_plot(data, battery_capacity_kwh, soc_max, soc_min,
                                  filename='daily_ieee_column'):
    """Create IEEE column-sized daily plot"""
    
    fig_width = 3.5
    fig_height = 6.0
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, fig_height))
    fig.subplots_adjust(hspace=0.25)
    
    colors = {
        'pv': '#FF6B00', 'export': '#38761D', 'import': '#990000',
        'soc': '#1F4E79', 'export_fill': '#90EE90', 'import_fill': '#FFB6C1',
    }
    
    # Calculate power limits for this day
    all_power_data = np.concatenate([
        data["AC_Power_kW_osm_total"].values,
        data['Grid_Power_KW'].clip(lower=0).values,
        data['Grid_Power_KW'].clip(upper=0).abs().values
    ])
    power_max = np.max(np.abs(all_power_data)) * 1.1
    
    # --- TOP PLOT ---
    ax1.plot(data.index, data["AC_Power_kW_osm_total"], 
             label="PV Power", color=colors['pv'], linewidth=1.0)
    
    export_power = data['Grid_Power_KW'].clip(lower=0)
    import_power = data['Grid_Power_KW'].clip(upper=0).abs()
    
    ax1.fill_between(data.index, export_power, alpha=0.3, 
                     color=colors['export_fill'], label='Grid Export')
    ax1.fill_between(data.index, -import_power, alpha=0.3, 
                     color=colors['import_fill'], label='Grid Import')
    
    ax1.plot(data.index, export_power, color=colors['export'], linewidth=0.8, linestyle='--')
    ax1.plot(data.index, -import_power, color=colors['import'], linewidth=0.8, linestyle='--')
    
    ax1.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)
    ax1.set_ylim([-30, power_max])
    ax1.set_ylabel("Power [kW]", fontname='Garamond', fontsize=9, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    
    # X-axis formatting for daily plot
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center', 
             fontname='Garamond', fontsize=8)
    ax1.set_xlabel("")
    
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), 
               ncol=3, framealpha=0.9, fontsize=7, fancybox=True, shadow=True)  # FIXED: fontsize not font-size
    
    # --- BOTTOM PLOT ---
    ax2.plot(data.index, data["Battery_SOC_kWh"], 
             label="Battery SOC", color=colors['soc'], linewidth=1.0)
    ax2.axhline(soc_max, color='red', linestyle='--', linewidth=0.8, alpha=0.8, label='SOC Limits')
    ax2.axhline(soc_min, color='red', linestyle='--', linewidth=0.8, alpha=0.8)
    
    ax2.set_ylim([0, battery_capacity_kwh * 1.05])
    ax2.set_ylabel("Battery SOC [kWh]", fontname='Garamond', fontsize=9, fontweight='bold')
    ax2.set_xlabel("Time of Day", fontname='Garamond', fontsize=9, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center', 
             fontname='Garamond', fontsize=8)
    
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), 
               ncol=2, framealpha=0.9, fontsize=7, fancybox=True, shadow=True)  # FIXED: fontsize not font-size
    
    xlim = (data.index.min(), data.index.max())
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    
    plt.tight_layout()
    
    plt.savefig(f'{filename}.pdf', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='pdf')
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='png')
    
    return fig

#-------------------------------------------------------------------------------------

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
    export_power = data['Grid_Power_KW'].clip(lower=0)
    import_power = data['Grid_Power_KW'].clip(upper=0).abs()
    
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
    ax2.plot(data.index, data["Battery_SOC_kWh"], 
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
    export_power = data['Grid_Power_KW'].clip(lower=0)
    import_power = data['Grid_Power_KW'].clip(upper=0).abs()
    
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
    
    ax2.plot(data.index, data["Battery_SOC_kWh"], 
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
    yearly_data['Grid_Power_KW'].clip(lower=0).values,
    yearly_data['Grid_Power_KW'].clip(upper=0).abs().values
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
