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

# === BATTERY SYSTEM EQUATIONS ===
battery_capacity_kwh = 150
battery_max_charge_kw = 15
battery_max_discharge_kw = 15
battery_efficiency = 0.90
dt = 5/60

soc_min = 0.10 * battery_capacity_kwh
soc_max = 0.95 * battery_capacity_kwh
battery_soc = 0.5 * battery_capacity_kwh

# === STRATEGIC LOAD PROFILE - PRIORITIZE 10AM-2PM FOR LOAD ===
def create_strategic_load(index, base=8, peak=34):
    """
    Strategic load profile that shifts most load to 10am-2pm
    and creates export opportunities from 6pm-9pm.
    """
    load = np.ones(len(index)) * base
    
    for i, dt in enumerate(index):
        h = dt.hour
        if 10 <= h < 14:     # Midday peak: 10am-2pm - HIGH demand for load shifting
            load[i] = base + 0.80*(peak - base)  # ~55 kW
        elif 18 <= h < 21:   # Evening export: 6pm-9pm - LOW demand for export
            load[i] = base + 0.20*(peak - base)  # ~22 kW
        elif 21 <= h or h < 6:  # Night: 9pm-6am - MEDIUM demand
            load[i] = base + 0.10*(peak - base) 
        elif 6 <= h < 10:    # Morning: 6am-10am - MEDIUM demand
            load[i] = base + 0.40*(peak - base)  # ~31 kW
        else:               # Afternoon: 2pm-6pm
            load[i] = base + 0.30*(peak - base)  # ~36 kW
    
    # Add random noise
    load += np.random.normal(0, 1.0, len(load))
    
    return np.clip(load, 8, peak)

df['Home_Load_kW'] = create_strategic_load(df.index)

# === STRATEGIC BATTERY MANAGEMENT PARAMETERS ===
enable_strategic_export = True

# Primary battery charging during PV hours (prioritize charging)
pv_charging_hours = [8, 9, 10, 11, 12, 13, 14, 15, 16]  # 8am-5pm

# Focused peak export: 6pm-9pm only (narrower window for concentrated export)
peak_export_hours = [18, 19, 20]  # 6pm-9pm

# Load shifting hours (when we want high consumption)
load_shift_hours = [10, 11, 12, 13]  # 10am-2pm

# More aggressive strategic export settings for 6pm-9pm
min_soc_for_export = 0.20  # Lower from 30% to 20% for more export opportunities
max_grid_export_from_battery = 15  # Increase from 12 kW to 15 kW for more export
export_net_power_threshold = 20    # Increase from 15 kW to 20 kW

# Import encouragement parameters for load shift hours
import_aggressiveness = 0.8  # More aggressive import during load shift hours

# === STRATEGIC BATTERY & GRID SIMULATION ===
grid_power, battery_power_list, battery_soc_list = [], [], []
battery_operation_mode = []

for i, pv_power in enumerate(df["AC_Power_kW_osm_total"]):
    load = df['Home_Load_kW'].iloc[i]
    current_hour = df.index[i].hour
    
    net_power = pv_power - load
    battery_power = 0
    operation_mode = "Idle"

    # === MODE 1: PRIORITY - BATTERY CHARGING DURING PV HOURS ===
    if (current_hour in pv_charging_hours and 
        battery_soc < soc_max and 
        pv_power > 0):  # Only when PV is producing
        
        operation_mode = "Charging from PV"
        
        # Calculate maximum possible charge from PV
        available_pv_power = max(0, pv_power - load)  # Excess PV after load
        charge_possible = min(available_pv_power, battery_max_charge_kw)
        energy_possible = (soc_max - battery_soc) / battery_efficiency
        charge_power_possible = energy_possible / dt
        charge_power = min(charge_possible, charge_power_possible)
        
        battery_soc += charge_power * dt * battery_efficiency
        net_power -= charge_power
        battery_power = charge_power

    # === MODE 2: STRATEGIC IMPORT FOR LOAD SHIFTING (10am-2pm) ===
    elif (current_hour in load_shift_hours and 
          battery_soc < soc_max and 
          net_power < 0):  # Only if we need power
        
        operation_mode = "Charging from Grid"
        
        # Calculate import power - be aggressive during load shift hours
        import_needed = min(-net_power, battery_max_charge_kw)
        energy_possible = (soc_max - battery_soc) / battery_efficiency
        charge_power_possible = energy_possible / dt
        
        # Use import aggressiveness factor
        strategic_import = min(import_needed, charge_power_possible) * import_aggressiveness
        charge_power = min(strategic_import, battery_max_charge_kw)
        
        battery_soc += charge_power * dt * battery_efficiency
        net_power -= charge_power  # More negative = more import
        battery_power = charge_power

    # === MODE 3: STRATEGIC BATTERY DISCHARGE TO GRID (6pm-9pm) ===
    elif (enable_strategic_export and 
          battery_soc > min_soc_for_export * battery_capacity_kwh and
          current_hour in peak_export_hours and
          battery_soc > soc_min):
        
        operation_mode = "Discharging to Grid"
        
        energy_available = (battery_soc - soc_min) * battery_efficiency
        discharge_power_possible = energy_available / dt
        
        # Calculate available export capacity
        available_for_export = min(discharge_power_possible, battery_max_discharge_kw)
        available_for_export = min(available_for_export, max_grid_export_from_battery)
        
        # Calculate TRUE surplus after supplying load
        current_load_deficit = max(-net_power, 0)  # How much load needs power
        
        if current_load_deficit > 0:
            # Supply load first, then export remainder
            load_supply_power = min(current_load_deficit, available_for_export)
            export_power = available_for_export - load_supply_power
        else:
            # No load deficit - all available power can be exported
            load_supply_power = 0
            export_power = available_for_export
        
        # Include only excess PV power (PV minus load)
        pv_export = max(pv_power - load, 0) if pv_power > 0 else 0
        
        total_export_power = export_power + pv_export
        total_discharge_power = load_supply_power + export_power
        
        # CORRECTED NET POWER CALCULATION:
        # Start from original net_power, then:
        # - Add battery power that supplies load (reduces import need)
        # - Add export power (increases export)
        net_power = net_power + load_supply_power + total_export_power
        
        # Update battery state
        battery_soc -= total_discharge_power * dt / battery_efficiency
        battery_power = -total_discharge_power

    # === MODE 4: BATTERY DISCHARGING TO SUPPLY OWN LOAD ===
    elif (net_power < 0 and battery_soc > soc_min and
          operation_mode != "Discharging to Grid" and
          current_hour not in peak_export_hours):  # Don't override strategic export
        
        operation_mode = "Discharging to Load"
        
        discharge_needed = min(-net_power, battery_max_discharge_kw)
        energy_available = (battery_soc - soc_min) * battery_efficiency
        discharge_power_possible = energy_available / dt
        discharge_power = min(discharge_needed, discharge_power_possible)
        
        battery_soc -= discharge_power * dt / battery_efficiency
        net_power += discharge_power
        battery_power = -discharge_power

    grid_power.append(net_power)
    battery_power_list.append(battery_power)
    battery_soc_list.append(battery_soc)
    battery_operation_mode.append(operation_mode)

df['Grid_Power_KW'] = grid_power
df['Battery_Power_kW'] = battery_power_list
df['Battery_SOC_kWh'] = battery_soc_list
df['Battery_Mode'] = battery_operation_mode

# === ENERGY CALCULATIONS ===
total_pv_energy = df["AC_Power_kW_osm_total"].sum() * dt
total_load_energy = df["Home_Load_kW"].sum() * dt
grid_export_energy = df[df['Grid_Power_KW'] > 0]['Grid_Power_KW'].sum() * dt
grid_import_energy = abs(df[df['Grid_Power_KW'] < 0]['Grid_Power_KW'].sum() * dt)
battery_charge_energy = df[df['Battery_Power_kW'] > 0]['Battery_Power_kW'].sum() * dt
battery_discharge_energy = abs(df[df['Battery_Power_kW'] < 0]['Battery_Power_kW'].sum() * dt)
self_consumption_ratio = (total_load_energy - grid_import_energy) / total_load_energy

# Strategic energy calculations
strategic_import_energy = df[df['Battery_Mode'] == 'Charging from Grid']['Battery_Power_kW'].sum() * dt
strategic_export_energy = df[df['Battery_Mode'] == 'Discharging to Grid']['Battery_Power_kW'].abs().sum() * dt
pv_charging_energy = df[df['Battery_Mode'] == 'Charging from PV']['Battery_Power_kW'].sum() * dt

# === ENHANCED PROFESSIONAL PLOT FUNCTION (9.5x8.5 inches) ===
def create_strategic_plot(data, time_period='daily', filename='strategic_plot'):
    """Professional plot with strategic energy management visualization."""
    
    # Create figure with requested size
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 7.5))
    fig.subplots_adjust(hspace=0.4)
    
    # Define strategic color scheme
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
        'load_shift': '#FF69B4',   # Pink for load shift periods
        'strategic_export': '#32CD32', # Lime green for strategic export periods
    }
    
    # --- TOP PLOT: POWER FLOWS WITH STRATEGIC HIGHLIGHTS ---
    
    # Calculate consistent y-axis limits
    all_power_data = np.concatenate([
        data["AC_Power_kW_osm_total"].values,
        data["Home_Load_kW"].values, 
        data["Battery_Power_kW"].values,
        data['Grid_Power_KW'].clip(lower=0).values,
        data['Grid_Power_KW'].clip(upper=0).abs().values
    ])
    
    power_max = np.max(np.abs(all_power_data)) * 1.1
    power_ylim = [-power_max, power_max]
    
    # Highlight strategic periods with background colors
    for hour_range, color, alpha, label in [
        (pv_charging_hours, colors['pv_charging'], 0.1, 'PV Charging Hours'),
        (load_shift_hours, colors['load_shift'], 0.15, 'Load Shift Hours'),
        (peak_export_hours, colors['strategic_export'], 0.15, 'Strategic Export Hours')
    ]:
        for date in pd.date_range(start=data.index.min().normalize(), 
                                 end=data.index.max().normalize(), freq='D'):
            for hour in hour_range:
                period_start = date + pd.Timedelta(hours=hour)
                period_end = period_start + pd.Timedelta(hours=1)
                ax1.axvspan(period_start, period_end, alpha=alpha, color=color,
                           label=label if date == data.index.min().normalize() and hour == hour_range[0] else "")
    
    # Plot power flows
    ax1.plot(data.index, data["AC_Power_kW_osm_total"], 
             label="PV Power", color=colors['pv'], linewidth=1.5)
    ax1.plot(data.index, data["Home_Load_kW"], 
             label="Load Power", color=colors['load'], linewidth=1.5)
    
    # Battery power with strategic modes
    battery_positive = data['Battery_Power_kW'].clip(lower=0)
    battery_negative = data['Battery_Power_kW'].clip(upper=0)
    
    ax1.plot(data.index, battery_positive, 
             label="Battery Charging", color=colors['battery_charge'], linewidth=1.5)
    ax1.plot(data.index, battery_negative, 
             label="Battery Discharging", color=colors['battery_discharge'], linewidth=1.5)
    
    # Grid power with area fills
    export_power = data['Grid_Power_KW'].clip(lower=0)
    import_power = data['Grid_Power_KW'].clip(upper=0).abs()
    
    ax1.fill_between(data.index, export_power, alpha=0.3, 
                     color=colors['export_fill'], label='Export Area')
    ax1.fill_between(data.index, -import_power, alpha=0.3, 
                     color=colors['import_fill'], label='Import Area')
    
    ax1.plot(data.index, export_power, 
             label="Export Power", color=colors['export'], linewidth=1.0, linestyle='--')
    ax1.plot(data.index, -import_power, 
             label="Import Power", color=colors['import'], linewidth=1.0, linestyle='--')
    
    ax1.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
    ax1.set_ylim(power_ylim)
    ax1.set_ylabel("Power [kW]", fontname='Garamond', fontsize=16, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # X-axis formatting
    if time_period == 'yearly':
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    elif time_period == 'monthly':
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
    else:
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%I %p'))
    
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center', fontname='Garamond', fontsize=14)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
    
    # Legend for top plot
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), 
               ncol=3, framealpha=0.9, fontsize=10, fancybox=True, shadow=True)
    
    # --- BOTTOM PLOT: BATTERY SOC AND MODES ---
    
    ax2.plot(data.index, data["Battery_SOC_kWh"], 
             label="Battery SOC", color=colors['soc'], linewidth=2.0)
    ax2.axhline(soc_max, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='Max SOC')
    ax2.axhline(soc_min, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='Min SOC')
    ax2.axhline(min_soc_for_export * battery_capacity_kwh, color='orange', linestyle=':', 
                linewidth=1.2, alpha=0.7, label='Strategic Export Threshold')
    
    ax2.set_ylim(0, battery_capacity_kwh * 1.05)
    ax2.set_ylabel("Battery SOC [kWh]", fontname='Garamond', fontsize=16, fontweight='bold')
    ax2.set_xlabel("Time", fontname='Garamond', fontsize=16, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # X-axis formatting (same as top)
    if time_period == 'yearly':
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    elif time_period == 'monthly':
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
    else:
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%I %p'))
    
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center', fontname='Garamond', fontsize=14)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
    
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), 
               ncol=3, framealpha=0.9, fontsize=10, fancybox=True, shadow=True)
    
    # Ensure same x-axis limits
    xlim = (data.index.min(), data.index.max())
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    
    plt.tight_layout()
    plt.savefig(f'{filename}.pdf', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='pdf')
    
    return fig

# === GENERATE STRATEGIC PLOTS ===
print("Generating strategic energy management plots...")

# Generate plots for key periods
plot_periods = [
    ('2024-01-01', '2024-01-07', 'January First Week', 'daily'),
    ('2024-07-01', '2024-07-07', 'July First Week', 'daily'),
    ('2024-12-01', '2024-12-07', 'December First Week', 'daily')
]

for start_date, end_date, period_name, time_period in plot_periods:
    plot_data = df[(df.index >= start_date) & (df.index < end_date)]
    if len(plot_data) > 0:
        filename = f"{period_name.lower().replace(' ', '_')}_strategic"
        fig = create_strategic_plot(plot_data, time_period, filename)
        plt.show()

# === STRATEGIC PERFORMANCE ANALYSIS ===
print("\n" + "="*60)
print("STRATEGIC ENERGY MANAGEMENT PERFORMANCE")
print("="*60)
print(f"Total PV Energy: {total_pv_energy:,.0f} kWh")
print(f"Total Load Energy: {total_load_energy:,.0f} kWh")
print(f"Grid Import: {grid_import_energy:,.0f} kWh")
print(f"Grid Export: {grid_export_energy:,.0f} kWh")
print(f"Self-Consumption Ratio: {self_consumption_ratio:.1%}")

print(f"\nBattery Charging from PV: {pv_charging_energy:.1f} kWh")
print(f"Strategic Import Energy (10am-2pm): {strategic_import_energy:.1f} kWh")
print(f"Strategic Export Energy (6pm-9pm): {strategic_export_energy:.1f} kWh")

# Time-based analysis
pv_hours_data = df[df.index.hour.isin(pv_charging_hours)]
load_shift_data = df[df.index.hour.isin(load_shift_hours)]
export_hours_data = df[df.index.hour.isin(peak_export_hours)]

battery_charge_pv_hours = pv_hours_data[pv_hours_data['Battery_Mode'] == 'Charging from PV']['Battery_Power_kW'].sum() * dt
import_during_load_shift = abs(load_shift_data[load_shift_data['Grid_Power_KW'] < 0]['Grid_Power_KW'].sum() * dt)
export_during_peak = export_hours_data[export_hours_data['Grid_Power_KW'] > 0]['Grid_Power_KW'].sum() * dt

print(f"\nBattery charging during PV hours (8am-5pm): {battery_charge_pv_hours:.1f} kWh")
print(f"Import during load shift hours (10am-2pm): {import_during_load_shift:.1f} kWh")
print(f"Export during strategic hours (6pm-9pm): {export_during_peak:.1f} kWh")

# Battery mode analysis
mode_counts = df['Battery_Mode'].value_counts()
print(f"\nBattery Operation Modes:")
for mode, count in mode_counts.items():
    percentage = (count / len(df)) * 100
    hours = count * dt
    print(f"  {mode}: {count:,} intervals ({percentage:.1f}%) - {hours:.1f} hours")

print("\nAll strategic energy management plots saved as 9.5x8.5 inch PDF files.")

