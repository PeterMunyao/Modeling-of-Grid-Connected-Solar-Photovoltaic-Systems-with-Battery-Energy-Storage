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
temp_coeff = -0.0045         # α_temp (°C⁻¹)
stc_irradiance = 1000        # I_STC (W/m²)
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

# === UPDATED LOAD PROFILE WITH YOUR SPECIFICATIONS ===
def create_realistic_load(index, base=8, peak=36):
    """
    Equation 12: Time-dependent load profile
    P_load(t) = P_base + f(t) × (P_peak - P_base) + ε
    where f(t) is time-dependent factor, ε ~ N(0,2.0)
    
    Adapted to maintain specifications:
    
    """
    load = np.ones(len(index)) * base
    
    for i, dt in enumerate(index):
        h = dt.hour
        if 7 <= h < 19:      # Daytime: 7am-7pm - 
            # Use 55% of peak capacity 
            load[i] = base + 0.55*(peak - base)
        else:               # Nighttime: 7pm-7am 
            # Use 25% of base load 
            load[i] = base * 0.25
    
    # Add random noise: ε ~ N(0, 1.0) - increased for realistic variation
    load += np.random.normal(0, 1.0, len(load))
    
    # Ensure load stays within 5-60 kW bounds (minimum 5kW, maximum 60kW)
    return np.clip(load, 10, peak)

df['Home_Load_kW'] = create_realistic_load(df.index)

# === GRID EXPORT STRATEGY PARAMETERS ===
enable_strategic_export = True
peak_export_hours = [14, 15, 16, 17, 18, 19, 20, 21]
min_soc_for_export = 0.30
max_grid_export_from_battery = 10
export_net_power_threshold = 10

# === BATTERY & GRID SIMULATION EQUATIONS ===
grid_power, battery_power_list, battery_soc_list = [], [], []
battery_operation_mode = []

# Reset battery SOC for each simulation
battery_soc = 0.5 * battery_capacity_kwh

print(f"DataFrame length: {len(df)}")  # Debug: check DataFrame size

for i, (timestamp, row) in enumerate(df.iterrows()):
    pv_power = row["AC_Power_kW_osm_total"]
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

print(f"Grid power list length: {len(grid_power)}")  # Debug: check list size

# Verify lengths match before assignment
if len(grid_power) != len(df):
    print(f"ERROR: Length mismatch! DataFrame: {len(df)}, Results: {len(grid_power)}")
    # If there's a mismatch, truncate or extend to match
    min_length = min(len(df), len(grid_power))
    df = df.iloc[:min_length]
    grid_power = grid_power[:min_length]
    battery_power_list = battery_power_list[:min_length]
    battery_soc_list = battery_soc_list[:min_length]
    battery_operation_mode = battery_operation_mode[:min_length]

# Now assign the columns
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

# === ENHANCED PROFESSIONAL IEEE PLOT FUNCTION ===
def create_professional_plot_ieee(data, time_period='daily', filename='plot'):
    """Professional IEEE-style plot with enhanced features and proper spacing."""
    
    # Create figure with increased height for better spacing
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))  # Increased height for spacing
    fig.subplots_adjust(hspace=0.4)  # Increased vertical spacing between subplots
    
    # Define unique colors for each component
    colors = {
        'pv': '#FF6B00',           # Bright orange for PV
        'load': '#C00000',         # Dark red for load
        'battery_charge': '#0047AB', # Blue for battery charging
        'battery_discharge': '#1E90FF', # Light blue for battery discharging
        'export': '#38761D',       # Green for export
        'import': '#990000',       # Dark red for import
        'soc': '#1F4E79',          # Navy blue for SOC
        'export_fill': '#90EE90',  # Light green for export area
        'import_fill': '#FFB6C1',  # Light red for import area
    }
    
    # --- TOP PLOT: POWER FLOWS WITH AREA FILLS ---
    
    # Calculate consistent y-axis limits for power flows
    all_power_data = np.concatenate([
        data["AC_Power_kW_osm_total"].values,
        data["Home_Load_kW"].values, 
        data["Battery_Power_kW"].values,
        data['Grid_Power_KW'].clip(lower=0).values,
        data['Grid_Power_KW'].clip(upper=0).abs().values
    ])
    
    # Set consistent power y-axis limits with 10% margin
    power_max = np.max(np.abs(all_power_data)) * 1.1
    power_ylim = [-power_max, power_max]
    
    # Plot PV Power
    ax1.plot(data.index, data["AC_Power_kW_osm_total"], 
             label="PV Power", color=colors['pv'], linewidth=1.2)
    
    # Plot Load Power
    ax1.plot(data.index, data["Home_Load_kW"], 
             label="Load Power", color=colors['load'], linewidth=1.2)
    
    # Plot Battery Power with separate colors for charge/discharge
    battery_positive = data['Battery_Power_kW'].clip(lower=0)  # Charging
    battery_negative = data['Battery_Power_kW'].clip(upper=0)  # Discharging
    
    ax1.plot(data.index, battery_positive, 
             label="Battery Charging", color=colors['battery_charge'], linewidth=1.2, linestyle='-')
    ax1.plot(data.index, battery_negative, 
             label="Battery Discharging", color=colors['battery_discharge'], linewidth=1.2, linestyle='-')
    
    # Highlight Grid Export and Import with area fills
    export_power = data['Grid_Power_KW'].clip(lower=0)
    import_power = data['Grid_Power_KW'].clip(upper=0).abs()
    
    # Area fills for export and import
    ax1.fill_between(data.index, export_power, alpha=0.4, 
                     color=colors['export_fill'], label='Export Area')
    ax1.fill_between(data.index, -import_power, alpha=0.4, 
                     color=colors['import_fill'], label='Import Area')
    
    # Line plots on top of area fills
    ax1.plot(data.index, export_power, 
             label="Export Power", color=colors['export'], linewidth=1.0, linestyle='--')
    ax1.plot(data.index, -import_power, 
             label="Import Power", color=colors['import'], linewidth=1.0, linestyle='--')
    
    ax1.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
    
    # Set consistent power y-axis limits
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
    else:  # daily or 4-day
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%I %p'))
    
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center', fontname='Garamond', fontsize=14)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
    
    # Power legend - positioned with ample space
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), 
               ncol=4, framealpha=0.9, fontsize=11, fancybox=True, shadow=True)
    
    # --- BOTTOM PLOT: BATTERY SOC WITH ALIGNED SCALE ---
    
    # Plot Battery SOC
    ax2.plot(data.index, data["Battery_SOC_kWh"], 
             label="Battery SOC", color=colors['soc'], linewidth=1.5)
    
    # SOC limits
    ax2.axhline(soc_max, color='red', linestyle='--', linewidth=1.2, alpha=0.8, label='Max SOC')
    ax2.axhline(soc_min, color='red', linestyle='--', linewidth=1.2, alpha=0.8, label='Min SOC')
    
    # Align SOC scale to start from zero at the same level as power y-axis
    soc_range = battery_capacity_kwh
    soc_ylim = [0, soc_range * 1.05]  # Force starting from zero
    
    # Calculate alignment to match power axis visual position
    power_zero_position = 0  # Power axis zero is at middle
    soc_zero_position = 0    # SOC axis zero is at bottom
    
    ax2.set_ylim(soc_ylim)
    ax2.set_ylabel("Battery SOC [kWh]", fontname='Garamond', fontsize=16, fontweight='bold')
    ax2.set_xlabel("Time", fontname='Garamond', fontsize=16, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # X-axis formatting (same as top plot)
    if time_period == 'yearly':
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    elif time_period == 'monthly':
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
    else:  # daily or 4-day
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%I %p'))
    
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center', fontname='Garamond', fontsize=14)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
    
    # SOC legend - positioned with ample space
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), 
               ncol=3, framealpha=0.9, fontsize=11, fancybox=True, shadow=True)
    
    # Ensure both plots have the same x-axis limits
    xlim = (data.index.min(), data.index.max())
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    
    plt.tight_layout()
    
    # Save as PDF with high quality
    plt.savefig(f'{filename}.pdf', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='pdf')
    
    return fig

# === GENERATE ENHANCED PLOTS WITH PROPER SPACING ===
print("Generating enhanced IEEE-style professional plots...")

# 1. Yearly plot
yearly_data = df[(df.index >= '2024-01-01') & (df.index < '2025-01-01')]
fig_yearly = create_professional_plot_ieee(yearly_data, 'yearly', 'yearly_ieee_enhanced')
plt.show()

# 2. Monthly plots
months = [
    ('2024-01-01', '2024-02-01', 'January'),
    ('2024-07-01', '2024-08-01', 'July'),
    ('2024-12-01', '2025-01-01', 'December')
]

for start_date, end_date, month_name in months:
    monthly_data = df[(df.index >= start_date) & (df.index < end_date)]
    if len(monthly_data) > 0:
        fig_monthly = create_professional_plot_ieee(monthly_data, 'monthly', f'{month_name.lower()}_ieee_enhanced')
        plt.show()

# 3. 4-day zoom plots
four_day_periods = [
    ('2024-01-01', '2024-01-05', 'January'),
    ('2024-05-01', '2024-05-05', 'May'),
    ('2024-07-01', '2024-07-05', 'July'),
    ('2024-12-01', '2024-12-05', 'December')
]

for start_date, end_date, month_name in four_day_periods:
    four_day_data = df[(df.index >= start_date) & (df.index < end_date)]
    if len(four_day_data) > 0:
        fig_daily = create_professional_plot_ieee(four_day_data, 'daily', f'{month_name.lower()}_4day_ieee_enhanced')
        plt.show()

# === SYSTEM STATISTICS ===
print("\n" + "="*60)
print("SYSTEM PERFORMANCE SUMMARY")
print("="*60)
print(f"Total PV Energy: {total_pv_energy:,.0f} kWh")
print(f"Total Load Energy: {total_load_energy:,.0f} kWh") 
print(f"Grid Export: {grid_export_energy:,.0f} kWh")
print(f"Grid Import: {grid_import_energy:,.0f} kWh")
print(f"Self-Consumption Ratio: {self_consumption_ratio:.1%}")
print(f"Battery Capacity: {battery_capacity_kwh} kWh")

# Load profile verification
daytime_load = df[(df.index.hour >= 7) & (df.index.hour < 19)]['Home_Load_kW'].mean()
nighttime_load = df[(df.index.hour < 7) | (df.index.hour >= 19)]['Home_Load_kW'].mean()
max_load = df['Home_Load_kW'].max()

print(f"\nLoad Profile Verification:")
print(f"Daytime average load (7am-7pm): {daytime_load:.1f} kW")
print(f"Nighttime average load (7pm-7am): {nighttime_load:.1f} kW")
print(f"Maximum load: {max_load:.1f} kW")

print("\nAll enhanced professional plots have been generated and saved as PDF files.")
print("Files saved:")
print("- yearly_ieee_enhanced.pdf")
print("- january_ieee_enhanced.pdf, july_ieee_enhanced.pdf, december_ieee_enhanced.pdf")
print("- january_4day_ieee_enhanced.pdf, may_4day_ieee_enhanced.pdf, july_4day_ieee_enhanced.pdf, december_4day_ieee_enhanced.pdf")

#-------------------------------------------------------------------------------------------------------------------------------------------

# === LOAD WEATHER DATA ===
file_path = 'csv_-29.815268_30.946439_fixed_23_0_PT5M.csv'
df = pd.read_csv(file_path)
df['period_end'] = pd.to_datetime(df['period_end'], utc=True)
df.set_index('period_end', inplace=True)

# Check what data we actually have
available_years = df.index.year.unique()
print(f"Available years: {sorted(available_years)}")

if 2024 in available_years:
    # Use 2024 data as requested
    df = df[(df.index >= '2024-01-01') & (df.index < '2025-01-01')]
    print(f"Using 2024 data: {len(df)} records")
else:
    # Use the latest available year
    latest_year = max(available_years)
    df = df[df.index.year == latest_year]
    print(f"2024 not available. Using {latest_year} data: {len(df)} records")

# Convert to SAST for local time analysis
df.index = df.index.tz_convert('Africa/Johannesburg')

print(f"Final data range: {df.index.min()} to {df.index.max()}")


#----------------------------------------------------------------------------------------------------------------------------

# === STRATEGIC EXPORT ANALYSIS ===
print("\n" + "="*50)
print("STRATEGIC EXPORT ANALYSIS")
print("="*50)

# Calculate strategic export statistics
strategic_export_mask = (df['Battery_Mode'] == 'Discharging to Grid')
strategic_export_energy = df[strategic_export_mask]['Battery_Power_kW'].abs().sum() * dt
strategic_export_intervals = len(df[strategic_export_mask])
strategic_export_hours = strategic_export_intervals * dt

# Peak hour analysis
peak_hour_data = df[df.index.hour.isin(peak_export_hours)]
peak_export_total = peak_hour_data[peak_hour_data['Grid_Power_KW'] > 0]['Grid_Power_KW'].sum() * dt

# Print results
print(f"Strategic Export Energy: {strategic_export_energy:.1f} kWh")
print(f"Strategic Export Time: {strategic_export_hours:.1f} hours ({strategic_export_intervals} intervals)")
print(f"Average Export Power: {strategic_export_energy/strategic_export_hours:.1f} kW" if strategic_export_hours > 0 else "No strategic export")

if strategic_export_energy > 0:
    battery_contribution = (strategic_export_energy / peak_export_total * 100) if peak_export_total > 0 else 0
    print(f"Battery Contribution to Peak Exports: {battery_contribution:.1f}%")
    
    # Strategic export timing analysis
    strategic_by_hour = df[strategic_export_mask].groupby(df[strategic_export_mask].index.hour)['Battery_Power_kW'].count()
    print(f"\nStrategic Export by Hour:")
    for hour, count in strategic_by_hour.items():
        print(f"  {hour:02d}:00 - {count:4d} intervals")
else:
    print("No strategic export occurred - check battery SOC and timing conditions")

print(f"\nPeak Export Hours: {peak_export_hours} ({(peak_export_hours[0])}:00 - {(peak_export_hours[-1]+1)}:00)")
print(f"Min SOC for Export: {min_soc_for_export*100:.0f}% ({min_soc_for_export * battery_capacity_kwh:.1f} kWh)")
print(f"Max Grid Export from Battery: {max_grid_export_from_battery} kW")

#---------------------------------------------------------------------------------------------------------------------------



# === IEEE SINGLE-COLUMN BATTERY MODES PLOTS ===
def create_ieee_battery_modes_plot(data, month_name, filename):
    """IEEE single-column battery modes plot with date numbers only."""
    
    # IEEE single-column width: 3.5 inches, height optimized for modes
    fig, ax = plt.subplots(figsize=(6.5, 5.5))  # IEEE single-column size
    
    # Define professional color scheme
    mode_colors = {
        'Idle': '#808080',              # Gray
        'Charging from PV': '#0047AB',  # Blue
        'Discharging to Load': '#1E90FF', # Light blue
        'Discharging to Grid': '#FF4500' # Orange-red
    }
    
    mode_order = ['Idle', 'Charging from PV', 'Discharging to Load', 'Discharging to Grid']
    mode_levels = {mode: i for i, mode in enumerate(mode_order)}
    
    # Create timeline plot for battery modes
    current_mode = None
    segment_start = None
    
    for timestamp, mode in zip(data.index, data['Battery_Mode']):
        if mode != current_mode:
            # End previous segment
            if current_mode is not None and segment_start is not None:
                segment_duration = (timestamp - segment_start).total_seconds() / 3600
                if segment_duration > 0.01:
                    ax.hlines(y=mode_levels[current_mode], 
                             xmin=segment_start, xmax=timestamp,
                             color=mode_colors[current_mode], 
                             linewidth=4,  # Thinner for small plot
                             label=current_mode if segment_start == data.index[0] else "")
            
            # Start new segment
            current_mode = mode
            segment_start = timestamp
    
    # Plot the last segment
    if current_mode is not None and segment_start is not None:
        ax.hlines(y=mode_levels[current_mode], 
                 xmin=segment_start, xmax=data.index[-1],
                 color=mode_colors[current_mode], 
                 linewidth=4)
    
    # Set up the categorical y-axis
    ax.set_yticks(list(mode_levels.values()))
    ax.set_yticklabels(list(mode_levels.keys()), fontname='Garamond', fontsize=8, fontweight='bold')
    ax.set_ylabel("Battery Mode", fontname='Garamond', fontsize=9, fontweight='bold')
    ax.set_xlabel("Day of Month", fontname='Garamond', fontsize=9, fontweight='bold')
    
    # Professional grid styling - NO grey background
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')
    # Removed: ax.set_facecolor('#f8f8f8')  # No grey background
    
    # X-axis formatting: DATE NUMBERS ONLY (1, 2, 3, ..., 31)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))  # Every 2 days to avoid crowding
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))   # Day number only
    
    # Remove x-axis rotation
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center', fontname='Garamond', fontsize=8)
    
    # Set y-axis limits with padding
    ax.set_ylim(-0.5, len(mode_order) - 0.5)
    
    # Professional legend - smaller for IEEE format
    handles = [plt.Line2D([0], [0], color=mode_colors[mode], linewidth=3, label=mode) 
               for mode in mode_order if mode in data['Battery_Mode'].values]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.4),
              ncol=2, framealpha=0.9, fontsize=7, fancybox=True, shadow=True)
    
    # NO TITLE (removed as requested)
    
    plt.tight_layout()
    
    # Save as high-quality PDF
    plt.savefig(f'{filename}.pdf', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='pdf')
    
    return fig

# === GENERATE IEEE MONTHLY BATTERY MODES PLOTS ===
print("Generating IEEE single-column monthly battery modes plots...")

# Monthly periods to plot
monthly_periods = [
    ('2024-01-01', '2024-02-01', 'January'),
    ('2024-07-01', '2024-08-01', 'July'),
    ('2024-12-01', '2025-01-01', 'December')
]

for start_date, end_date, month_name in monthly_periods:
    monthly_data = df[(df.index >= start_date) & (df.index < end_date)]
    if len(monthly_data) > 0:
        print(f"\nGenerating {month_name} IEEE battery modes plot...")
        
        # Create and save plot
        filename = f"{month_name.lower()}_battery_modes_ieee"
        fig = create_ieee_battery_modes_plot(monthly_data, month_name, filename)
        plt.show()

# === STRATEGIC EXPORT DEBUGGING ===
print("\n" + "="*50)
print("STRATEGIC EXPORT DEBUGGING")
print("="*50)

# Quick analysis of why no Discharging to Grid
for month_name, (start_date, end_date, _) in zip(['January', 'July', 'December'], monthly_periods):
    monthly_data = df[(df.index >= start_date) & (df.index < end_date)]
    peak_hours = monthly_data[monthly_data.index.hour.isin([14, 15, 16, 17, 18, 19, 20, 21])]
    
    # Check conditions
    soc_ok = peak_hours['Battery_SOC_kWh'] > min_soc_for_export * battery_capacity_kwh
    net_power_ok = peak_hours['Grid_Power_KW'] <= export_net_power_threshold
    suitable = peak_hours[soc_ok & net_power_ok]
    
    print(f"{month_name}: {len(suitable)}/{len(peak_hours)} peak intervals met conditions")
    if len(suitable) > 0:
        avg_soc = suitable['Battery_SOC_kWh'].mean()
        avg_net_power = suitable['Grid_Power_KW'].mean()
        print(f"  Avg SOC: {avg_soc:.1f} kWh, Avg Net Power: {avg_net_power:.1f} kW")

print("\nAll IEEE single-column battery modes plots saved as PDF files.")
print("Files created:")
for month_name in ['january', 'july', 'december']:
    print(f"- {month_name}_battery_modes_ieee.pdf")


#-----------------------------------------------------------------------------------------------------------------

# === IEEE SINGLE-COLUMN BATTERY MODES PLOTS - FULL YEAR (MARKERS ONLY FOR GRID DISCHARGE) ===
def create_ieee_battery_modes_plot_full_year_markers_only(data, period_name, filename):
    """IEEE single-column battery modes plot for full year with markers only for grid discharge."""
    
    # IEEE single-column width: 3.5 inches, height optimized for modes
    fig, ax = plt.subplots(figsize=(6.5, 5.5))  # IEEE single-column size
    
    # Define professional color scheme
    mode_colors = {
        'Idle': '#808080',              # Gray
        'Charging from PV': '#0047AB',  # Blue
        'Discharging to Load': '#1E90FF', # Light blue
        'Discharging to Grid': '#FF0000'  # Red (for legend only)
    }
    
    # Line widths for non-grid modes
    mode_linewidths = {
        'Idle': 3.5,
        'Charging from PV': 3.5,
        'Discharging to Load': 3.5,
        'Discharging to Grid': 0  # No line for grid discharge
    }
    
    mode_order = ['Idle', 'Charging from PV', 'Discharging to Load', 'Discharging to Grid']
    mode_levels = {mode: i for i, mode in enumerate(mode_order)}
    
    # First, let's analyze the data to see what we're dealing with
    grid_discharge_data = data[data['Battery_Mode'] == 'Discharging to Grid']
    print(f"\nGrid Discharge Analysis:")
    print(f"  Total 'Discharging to Grid' intervals: {len(grid_discharge_data)}")
    if len(grid_discharge_data) > 0:
        print(f"  First occurrence: {grid_discharge_data.index[0]}")
        print(f"  Last occurrence: {grid_discharge_data.index[-1]}")
        print(f"  Average duration: {(grid_discharge_data.index[-1] - grid_discharge_data.index[0]).total_seconds() / 3600:.2f} hours")
    
    # Create timeline plot for battery modes - SKIP GRID DISCHARGE LINES
    current_mode = None
    segment_start = None
    
    for timestamp, mode in zip(data.index, data['Battery_Mode']):
        if mode != current_mode:
            # End previous segment
            if current_mode is not None and segment_start is not None:
                segment_duration = (timestamp - segment_start).total_seconds() / 3600
                if segment_duration > 0.01 and current_mode != 'Discharging to Grid':  # SKIP GRID DISCHARGE LINES
                    linewidth = mode_linewidths[current_mode]
                    ax.hlines(y=mode_levels[current_mode], 
                             xmin=segment_start, xmax=timestamp,
                             color=mode_colors[current_mode], 
                             linewidth=linewidth,
                             label=current_mode if segment_start == data.index[0] else "")
            
            # Start new segment
            current_mode = mode
            segment_start = timestamp
    
    # Plot the last segment - SKIP GRID DISCHARGE LINES
    if current_mode is not None and segment_start is not None and current_mode != 'Discharging to Grid':
        linewidth = mode_linewidths[current_mode]
        ax.hlines(y=mode_levels[current_mode], 
                 xmin=segment_start, xmax=data.index[-1],
                 color=mode_colors[current_mode], 
                 linewidth=linewidth)
    
    # MARKER POINTS ONLY FOR GRID DISCHARGE
    grid_discharge_times = data[data['Battery_Mode'] == 'Discharging to Grid'].index
    if len(grid_discharge_times) > 0:
        # Plot individual points for each grid discharge interval - MARKERS ONLY
        y_pos = mode_levels['Discharging to Grid']
        ax.scatter(grid_discharge_times, [y_pos] * len(grid_discharge_times), 
                  color='red', s=5, zorder=5, alpha=0.9,  # Changed to red color, increased size
                  edgecolors='darkred', linewidths=0.8,
                  marker='o', label='Discharging to Grid')
    
    # Set up the categorical y-axis
    ax.set_yticks(list(mode_levels.values()))
    ax.set_yticklabels(list(mode_levels.keys()), fontname='Garamond', fontsize=10, fontweight='bold')
    ax.set_ylabel("Battery Mode", fontname='Garamond', fontsize=11, fontweight='bold')
    ax.set_xlabel("Month", fontname='Garamond', fontsize=11, fontweight='bold')
    
    # Professional grid styling
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')
    
    # X-axis formatting: MONTHLY TICKS for full year
    ax.xaxis.set_major_locator(mdates.MonthLocator())  # Every month
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))   # Month abbreviations
    
    # Remove x-axis rotation
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center', fontname='Garamond', fontsize=8)
    
    # Set y-axis limits with extra padding for markers
    ax.set_ylim(-0.7, len(mode_order) - 0.3)
    
    # Set x-axis limits to cover full year
    ax.set_xlim(data.index.min(), data.index.max())
    
    # Simplified legend - markers only for grid discharge
    handles = []
    for mode in mode_order:
        if mode in data['Battery_Mode'].values:
            if mode == 'Discharging to Grid':
                # Marker only for grid discharge
                marker_handle = plt.Line2D([0], [0], marker='o', color='red', markersize=6, 
                                         linestyle='None', markeredgecolor='darkred', 
                                         label='Discharging to Grid')
                handles.append(marker_handle)
            else:
                # Line for other modes
                handles.append(plt.Line2D([0], [0], color=mode_colors[mode], linewidth=3, label=mode))
    
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.5),
              ncol=2, framealpha=0.95, fontsize=9, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    # Save as high-quality PDF
    plt.savefig(f'{filename}.pdf', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='pdf')
    
    return fig

# === GENERATE IEEE FULL YEAR BATTERY MODES PLOT WITH MARKERS ONLY ===
print("Generating IEEE single-column FULL YEAR battery modes plot with MARKERS ONLY for grid discharge...")

# Create full year data (January 1, 2024 to December 31, 2024)
full_year_data = df[(df.index >= '2024-01-01') & (df.index < '2025-01-01')]

if len(full_year_data) > 0:
    print(f"\nGenerating FULL YEAR IEEE battery modes plot (markers only for grid discharge)...")
    print(f"Data range: {full_year_data.index.min()} to {full_year_data.index.max()}")
    print(f"Total data points: {len(full_year_data)}")
    
    # Create and save plot with markers only
    filename = "full_year_battery_modes_ieee_markers_only"
    fig = create_ieee_battery_modes_plot_full_year_markers_only(full_year_data, "Full Year 2024", filename)
    plt.show()
  

# === GRID DISCHARGE DETAILED ANALYSIS ===
grid_discharge = full_year_data[full_year_data['Battery_Mode'] == 'Discharging to Grid']
if len(grid_discharge) > 0:
    print(f"\n=== GRID DISCHARGE DETAILED ANALYSIS ===")
    print(f"Total grid discharge energy: {abs(grid_discharge['Battery_Power_kW'].sum() * (5/60)):.2f} kWh")
    
    # CORRECTED POWER CALCULATIONS:
    # For discharge, battery power should be NEGATIVE, so we look for the most negative value
    discharge_power_values = grid_discharge['Battery_Power_kW']
    
    # Maximum discharge power (most negative value converted to positive)
    max_discharge_power = abs(discharge_power_values.min())
    
    # Average discharge power (mean of absolute values of negative powers)
    avg_discharge_power = abs(discharge_power_values[discharge_power_values < 0].mean())
    
    print(f"Average discharge power: {avg_discharge_power:.2f} kW")
    print(f"Maximum discharge power: {max_discharge_power:.2f} kW")
    
    # Monthly breakdown of grid discharge
    print(f"\nGrid discharge by month:")
    for month in range(1, 13):
        month_data = grid_discharge[grid_discharge.index.month == month]
        if len(month_data) > 0:
            energy = abs(month_data['Battery_Power_kW'].sum() * (5/60))
            print(f"  Month {month}: {len(month_data)} intervals, {energy:.2f} kWh")

print("\nFull year IEEE battery modes plot (markers only) saved as PDF file.")
print("File created: full_year_battery_modes_ieee_markers_only.pdf")

#----------------------------------------------------------------------------------------



