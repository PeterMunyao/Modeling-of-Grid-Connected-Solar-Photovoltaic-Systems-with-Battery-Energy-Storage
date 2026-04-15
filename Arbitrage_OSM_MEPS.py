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

# === STRATEGIC LOAD PROFILE ===
def create_strategic_load(index, base=5, peak=33):
    """
    Strategic load profile that encourages import during 9am-3pm
    and creates demand peaks for battery discharge.
    """
    load = np.ones(len(index)) * base
    
    for i, dt in enumerate(index):
        h = dt.hour
        if 18 <= h < 22:    # Evening peak: 6pm-10pm - 
            load[i] = base + 0.20*(peak - base)  #
        elif 22 <= h or h < 6:  # Night: 10pm-6am - 
            load[i] = base + 0.20*(peak - base)  # 
        elif 9 <= h < 15:   # Midday: 9am-3pm
            load[i] = base + 0.80*(peak - base)  # 
        else:               # Other hours: 6am-9am, 3pm-6pm
            load[i] = base + 0.25*(peak - base)  # 
    
    # Add random noise
    load += np.random.normal(0, 1.0, len(load))
    
    return np.clip(load, 5, peak)

df['Home_Load_kW'] = create_strategic_load(df.index)

# === STRATEGIC BATTERY MANAGEMENT PARAMETERS ===
enable_strategic_export = True
# Focused peak export: 6pm-10pm only
peak_export_hours = [18, 19, 20, 21]  # 6pm-10pm
# Import encouragement: 9am-3pm
import_encouragement_hours = [9, 10, 11, 12, 13, 14]  # 9am-3pm

# More aggressive strategic export settings
min_soc_for_export = 0.20  # Lower from 30% to 20% for more export opportunities
max_grid_export_from_battery = 15  # Increase from 10 kW to 12 kW
export_net_power_threshold = 15    # Increase from 10 kW to 15 kW

# Import encouragement parameters
import_aggressiveness = 0.7  # How aggressively to import during 9am-3pm (0-1)

# === STRATEGIC BATTERY & GRID SIMULATION ===
# Initialize lists with proper length
grid_power, battery_power_list, battery_soc_list = [], [], []
battery_operation_mode = []

# Initialize battery state
battery_soc = 0.5 * battery_capacity_kwh  # Start at 50% SOC

print(f"Starting battery simulation for {len(df)} timesteps...")

for i, (timestamp, row) in enumerate(df.iterrows()):
    pv_power = row["AC_Power_kW_osm_total"]
    load = row['Home_Load_kW']
    current_hour = timestamp.hour
    
    net_power = pv_power - load
    battery_power = 0
    operation_mode = "Idle"

    # === MODE 1: STRATEGIC IMPORT ENCOURAGEMENT (9am-3pm) ===
    if (current_hour in import_encouragement_hours and 
        battery_soc < soc_max and 
        net_power < 0):  # Only if we need power
        
        operation_mode = "Charging from Grid"
        
        # Calculate import power - be more aggressive during import hours
        import_needed = min(-net_power, battery_max_charge_kw)
        energy_possible = (soc_max - battery_soc) / battery_efficiency
        charge_power_possible = energy_possible / dt
        
        # Use import aggressiveness factor
        strategic_import = min(import_needed, charge_power_possible) * import_aggressiveness
        charge_power = min(strategic_import, battery_max_charge_kw)
        
        battery_soc += charge_power * dt * battery_efficiency
        net_power -= charge_power  # More negative = more import
        battery_power = charge_power

    # === MODE 2: BATTERY CHARGING FROM EXCESS PV ===
    elif (net_power > 0 and battery_soc < soc_max and
          operation_mode != "Charging from Grid"):  # Don't override strategic import
        
        operation_mode = "Charging from PV"
        
        charge_possible = min(net_power, battery_max_charge_kw)
        energy_possible = (soc_max - battery_soc) / battery_efficiency
        charge_power_possible = energy_possible / dt
        charge_power = min(charge_possible, charge_power_possible)
        
        battery_soc += charge_power * dt * battery_efficiency
        net_power -= charge_power
        battery_power = charge_power

    # === MODE 3: STRATEGIC BATTERY DISCHARGE TO GRID (6pm-10pm) ===
    elif (enable_strategic_export and 
          battery_soc > min_soc_for_export * battery_capacity_kwh and
          current_hour in peak_export_hours and
          battery_soc > soc_min):
        
        operation_mode = "Discharging to Grid"
        
        energy_available = (battery_soc - soc_min) * battery_efficiency
        discharge_power_possible = energy_available / dt
        
        # Calculate TRUE export capacity after supplying load
        load_deficit = max(-net_power, 0)  # How much load needs supply
        available_for_export = min(discharge_power_possible, battery_max_discharge_kw)
        
        # Supply load first, then export remainder
        if load_deficit > 0:
            load_supply = min(load_deficit, available_for_export)
            export_power = available_for_export - load_supply
        else:
            export_power = available_for_export  # Already have surplus
        
        # Limit export to grid capacity
        export_power = min(export_power, max_grid_export_from_battery)
        
        battery_soc -= (load_supply + export_power) * dt / battery_efficiency
        net_power += load_supply  # Reduce import need
        net_power += export_power  # Add export
        battery_power = -(load_supply + export_power)

    # === MODE 4: BATTERY DISCHARGING TO SUPPLY OWN LOAD ===
    elif (net_power < 0 and battery_soc > soc_min and
          operation_mode != "Discharging to Grid"):  # Don't override strategic export
        
        operation_mode = "Discharging to Load"
        
        discharge_needed = min(-net_power, battery_max_discharge_kw)
        energy_available = (battery_soc - soc_min) * battery_efficiency
        discharge_power_possible = energy_available / dt
        discharge_power = min(discharge_needed, discharge_power_possible)
        
        battery_soc -= discharge_power * dt / battery_efficiency
        net_power += discharge_power
        battery_power = -discharge_power

    # Ensure battery SOC stays within bounds
    battery_soc = max(soc_min, min(battery_soc, soc_max))
    
    # Append results
    grid_power.append(net_power)
    battery_power_list.append(battery_power)
    battery_soc_list.append(battery_soc)
    battery_operation_mode.append(operation_mode)
    
    # Progress indicator for long simulations
    if i % 10000 == 0:
        print(f"Processed {i}/{len(df)} timesteps...")

print(f"Battery simulation completed. Processed {len(grid_power)} timesteps.")

# Verify lengths match
print(f"DataFrame length: {len(df)}")
print(f"Grid power list length: {len(grid_power)}")
print(f"Battery power list length: {len(battery_power_list)}")
print(f"Battery SOC list length: {len(battery_soc_list)}")

# Only assign if lengths match
if len(grid_power) == len(df):
    df['Grid_Power_KW'] = grid_power
    df['Battery_Power_kW'] = battery_power_list
    df['Battery_SOC_kWh'] = battery_soc_list
    df['Battery_Mode'] = battery_operation_mode
    print("Successfully assigned battery simulation results to DataFrame.")
else:
    print(f"ERROR: Length mismatch! DataFrame: {len(df)}, Results: {len(grid_power)}")

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

# === CORRECTED PROFESSIONAL PLOT FUNCTION (9.5x8.5 inches) ===
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
        'strategic_import': '#FFA500', # Orange for strategic import periods
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
    
    # Create custom legend handles for strategic periods FIRST
    strategic_import_patch = plt.Rectangle((0,0), 1, 1, fc=colors['strategic_import'], alpha=0.15, 
                                          label='Strategic Import Hours (9am-3pm)')
    strategic_export_patch = plt.Rectangle((0,0), 1, 1, fc=colors['strategic_export'], alpha=0.15, 
                                          label='Strategic Export Hours (6pm-10pm)')
    
    # Highlight strategic periods with background colors (NO LABELS in loop)
    for hour_range, color, alpha in [
        (import_encouragement_hours, colors['strategic_import'], 0.15),
        (peak_export_hours, colors['strategic_export'], 0.15)
    ]:
        for date in pd.date_range(start=data.index.min().normalize(), 
                                 end=data.index.max().normalize(), freq='D'):
            for hour in hour_range:
                period_start = date + pd.Timedelta(hours=hour)
                period_end = period_start + pd.Timedelta(hours=1)
                ax1.axvspan(period_start, period_end, alpha=alpha, color=color)
    
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
    
    # Create combined legend with power elements + strategic period patches
    power_handles, power_labels = ax1.get_legend_handles_labels()
    
    # Add strategic period patches to the legend
    all_handles = power_handles + [strategic_import_patch, strategic_export_patch]
    all_labels = power_labels + ['Strategic Import Hours (9am-3pm)', 'Strategic Export Hours (6pm-10pm)']
    
    ax1.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, -0.3), 
               ncol=3, framealpha=0.9, fontsize=9, fancybox=True, shadow=True)
    
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

# === GENERATE, DISPLAY AND SAVE STRATEGIC PLOTS ===
print("Generating and displaying strategic energy management plots...")

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
        print(f"Creating {filename}.pdf...")
        fig = create_strategic_plot(plot_data, time_period, filename)
        plt.show()  # Display the plot
        plt.close(fig)  # Close the figure to free memory

# Generate additional strategic plots
print("\nGenerating additional strategic plots...")

# Yearly overview
yearly_data = df[(df.index >= '2024-01-01') & (df.index < '2025-01-01')]
if len(yearly_data) > 0:
    print("Creating yearly_strategic_overview.pdf...")
    fig = create_strategic_plot(yearly_data, 'yearly', 'yearly_strategic_overview')
    plt.show()  # Display the plot
    plt.close(fig)

# Monthly plots
monthly_periods = [
    ('2024-01-01', '2024-02-01', 'January'),
    ('2024-07-01', '2024-08-01', 'July'),
    ('2024-12-01', '2025-01-01', 'December')
]

for start_date, end_date, month_name in monthly_periods:
    monthly_data = df[(df.index >= start_date) & (df.index < end_date)]
    if len(monthly_data) > 0:
        filename = f"{month_name.lower()}_monthly_strategic"
        print(f"Creating {filename}.pdf...")
        fig = create_strategic_plot(monthly_data, 'monthly', filename)
        plt.show()  # Display the plot
        plt.close(fig)

# 4-day detailed plots
four_day_periods = [
    ('2024-01-01', '2024-01-05', 'January_4day'),
    ('2024-04-01', '2024-04-05', 'April_4day'),
    ('2024-07-01', '2024-07-05', 'July_4day'),
    ('2024-10-01', '2024-10-05', 'October_4day')
]

for start_date, end_date, period_name in four_day_periods:
    four_day_data = df[(df.index >= start_date) & (df.index < end_date)]
    if len(four_day_data) > 0:
        filename = f"{period_name.lower()}_strategic"
        print(f"Creating {filename}.pdf...")
        fig = create_strategic_plot(four_day_data, 'daily', filename)
        plt.show()  # Display the plot
        plt.close(fig)

print("\n" + "="*60)
print("PLOT GENERATION COMPLETE")
print("="*60)
print("All strategic energy management plots have been saved as PDF files:")
print("\nWeekly Plots:")
print("- january_first_week_strategic.pdf")
print("- july_first_week_strategic.pdf")
print("- december_first_week_strategic.pdf")

print("\nOverview Plots:")
print("- yearly_strategic_overview.pdf")

print("\nMonthly Plots:")
print("- january_monthly_strategic.pdf")
print("- july_monthly_strategic.pdf")
print("- december_monthly_strategic.pdf")

print("\nDetailed 4-Day Plots:")
print("- january_4day_strategic.pdf")
print("- april_4day_strategic.pdf")
print("- july_4day_strategic.pdf")
print("- october_4day_strategic.pdf")

print(f"\nTotal plots generated: {3 + 1 + 3 + 4} PDF files")

# === GENERATE REQUESTED STRATEGIC PLOTS ===
print("Generating requested strategic energy management plots...")

# 1. Whole Year 2024 Plot
print("\n1. Creating Whole Year 2024 plot...")
yearly_data = df[(df.index >= '2024-01-01') & (df.index < '2025-01-01')]
if len(yearly_data) > 0:
    fig = create_strategic_plot(yearly_data, 'yearly', 'whole_year_2024_strategic')
    plt.show()
    plt.close(fig)

# 2. January to June 2024 Plot
print("\n2. Creating January-June 2024 plot...")
jan_june_data = df[(df.index >= '2024-01-01') & (df.index < '2024-07-01')]
if len(jan_june_data) > 0:
    fig = create_strategic_plot(jan_june_data, 'monthly', 'jan_to_june_2024_strategic')
    plt.show()
    plt.close(fig)

# 3. July to December 2024 Plot
print("\n3. Creating July-December 2024 plot...")
july_dec_data = df[(df.index >= '2024-07-01') & (df.index < '2025-01-01')]
if len(july_dec_data) > 0:
    fig = create_strategic_plot(july_dec_data, 'monthly', 'july_to_dec_2024_strategic')
    plt.show()
    plt.close(fig)

# === SEASONAL ANALYSIS ===
print("\n" + "="*60)
print("SEASONAL PERFORMANCE ANALYSIS")
print("="*60)

# Calculate seasonal statistics
if len(jan_june_data) > 0:
    jan_june_pv = jan_june_data["AC_Power_kW_osm_total"].sum() * dt
    jan_june_load = jan_june_data["Home_Load_kW"].sum() * dt
    jan_june_export = jan_june_data[jan_june_data['Grid_Power_KW'] > 0]['Grid_Power_KW'].sum() * dt
    jan_june_import = abs(jan_june_data[jan_june_data['Grid_Power_KW'] < 0]['Grid_Power_KW'].sum() * dt)
    
    print(f"January-June 2024:")
    print(f"  PV Generation: {jan_june_pv:,.0f} kWh")
    print(f"  Load Consumption: {jan_june_load:,.0f} kWh")
    print(f"  Grid Export: {jan_june_export:,.0f} kWh")
    print(f"  Grid Import: {jan_june_import:,.0f} kWh")
    print(f"  Net Export: {jan_june_export - jan_june_import:,.0f} kWh")

if len(july_dec_data) > 0:
    july_dec_pv = july_dec_data["AC_Power_kW_osm_total"].sum() * dt
    july_dec_load = july_dec_data["Home_Load_kW"].sum() * dt
    july_dec_export = july_dec_data[july_dec_data['Grid_Power_KW'] > 0]['Grid_Power_KW'].sum() * dt
    july_dec_import = abs(july_dec_data[july_dec_data['Grid_Power_KW'] < 0]['Grid_Power_KW'].sum() * dt)
    
    print(f"\nJuly-December 2024:")
    print(f"  PV Generation: {july_dec_pv:,.0f} kWh")
    print(f"  Load Consumption: {july_dec_load:,.0f} kWh")
    print(f"  Grid Export: {july_dec_export:,.0f} kWh")
    print(f"  Grid Import: {july_dec_import:,.0f} kWh")
    print(f"  Net Export: {july_dec_export - july_dec_import:,.0f} kWh")

# Strategic operation analysis by season
if len(jan_june_data) > 0:
    jan_june_strategic_import = jan_june_data[jan_june_data['Battery_Mode'] == 'Charging from Grid']['Battery_Power_kW'].sum() * dt
    jan_june_strategic_export = jan_june_data[jan_june_data['Battery_Mode'] == 'Discharging to Grid']['Battery_Power_kW'].abs().sum() * dt
    
    print(f"\nJanuary-June Strategic Operations:")
    print(f"  Strategic Import (9am-3pm): {jan_june_strategic_import:.1f} kWh")
    print(f"  Strategic Export (6pm-10pm): {jan_june_strategic_export:.1f} kWh")

if len(july_dec_data) > 0:
    july_dec_strategic_import = july_dec_data[july_dec_data['Battery_Mode'] == 'Charging from Grid']['Battery_Power_kW'].sum() * dt
    july_dec_strategic_export = july_dec_data[july_dec_data['Battery_Mode'] == 'Discharging to Grid']['Battery_Power_kW'].abs().sum() * dt
    
    print(f"\nJuly-December Strategic Operations:")
    print(f"  Strategic Import (9am-3pm): {july_dec_strategic_import:.1f} kWh")
    print(f"  Strategic Export (6pm-10pm): {july_dec_strategic_export:.1f} kWh")

print("\n" + "="*60)
print("PLOT GENERATION COMPLETE")
print("="*60)
print("Three main strategic plots generated:")
print("1. whole_year_2024_strategic.pdf - Complete year overview")
print("2. jan_to_june_2024_strategic.pdf - First half year (Summer/Autumn)")
print("3. july_to_dec_2024_strategic.pdf - Second half year (Winter/Spring)")
print("\nThese plots show:")
print("- Power flows (PV, Load, Battery, Grid)")
print("- Strategic import hours (9am-3pm) - Orange shading")
print("- Strategic export hours (6pm-10pm) - Green shading")
print("- Battery state of charge dynamics")

#------------------------------------------------------------------------------------------------------


# === EXTRACT AND PLOT APRIL DAY 1 AND DAY 3 ONLY (SEPARATE SUBPLOTS) ===
april_day_1 = df[(df.index >= '2024-04-01') & (df.index < '2024-04-02')]
april_day_3 = df[(df.index >= '2024-04-03') & (df.index < '2024-04-04')]

# Create figure with two columns for day 1 and day 3
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))
fig.subplots_adjust(hspace=0.3, wspace=0.15)

# Define colors including strategic shading colors
colors = {
    'pv': '#FF6B00', 'load': '#C00000', 'battery_charge': '#0047AB',
    'battery_discharge': '#1E90FF', 'export': '#38761D', 'import': '#990000',
    'soc': '#1F4E79', 'export_fill': '#90EE90', 'import_fill': '#FFB6C1',
    'strategic_import': '#FFA500',  # Orange for strategic import hours
    'strategic_export': '#32CD32',  # Green for strategic export hours
    'soc_limit': '#FF0000',  # Red for SOC limits
}

# Strategic hours parameters
import_encouragement_hours = [9, 10, 11, 12, 13, 14]  # 9am-3pm
peak_export_hours = [18, 19, 20, 21]  # 6pm-10pm

# DAY 1 - POWER FLOWS
power_ylim = [-30, 100]

# Add strategic shading for Day 1
for hour_range, color, alpha, label in [
    (import_encouragement_hours, colors['strategic_import'], 0.15, 'Strategic Import (9am-3pm)'),
    (peak_export_hours, colors['strategic_export'], 0.15, 'Strategic Export (6pm-10pm)')
]:
    for hour in hour_range:
        period_start = april_day_1.index[0].normalize() + pd.Timedelta(hours=hour)
        period_end = period_start + pd.Timedelta(hours=1)
        ax1.axvspan(period_start, period_end, alpha=alpha, color=color)

# Plot components for Day 1
ax1.plot(april_day_1.index, april_day_1["AC_Power_kW_osm_total"], 
         label="PV Power", color=colors['pv'], linewidth=1.5)
ax1.plot(april_day_1.index, april_day_1["Home_Load_kW"], 
         label="Load Power", color=colors['load'], linewidth=1.5)

# Battery power Day 1
battery_positive_1 = april_day_1['Battery_Power_kW'].clip(lower=0)
battery_negative_1 = april_day_1['Battery_Power_kW'].clip(upper=0)
ax1.plot(april_day_1.index, battery_positive_1, 
         label="Battery Charging", color=colors['battery_charge'], linewidth=1.5)
ax1.plot(april_day_1.index, battery_negative_1, 
         label="Battery Discharging", color=colors['battery_discharge'], linewidth=1.5)

# Grid power with area fills Day 1
export_power_1 = april_day_1['Grid_Power_KW'].clip(lower=0)
import_power_1 = april_day_1['Grid_Power_KW'].clip(upper=0).abs()
ax1.fill_between(april_day_1.index, export_power_1, alpha=0.4, color=colors['export_fill'])
ax1.fill_between(april_day_1.index, -import_power_1, alpha=0.4, color=colors['import_fill'])
ax1.plot(april_day_1.index, export_power_1, label="Export", color=colors['export'], linewidth=1.5, linestyle='--')
ax1.plot(april_day_1.index, -import_power_1, label="Import", color=colors['import'], linewidth=1.5, linestyle='--')

ax1.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
ax1.set_ylim(power_ylim)
ax1.set_ylabel("Power [kW]", fontname='Garamond', fontsize=18, fontweight='bold')

# Configure grids
ax1.grid(True, which='major', linestyle='-', alpha=0.3, linewidth=0.8)
ax1.grid(True, which='minor', linestyle='--', alpha=0.2, linewidth=0.5)

# X-axis formatting Day 1
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:00'))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center', fontname='Garamond', fontsize=15)

# DAY 3 - POWER FLOWS
# Add strategic shading for Day 3
for hour_range, color, alpha, label in [
    (import_encouragement_hours, colors['strategic_import'], 0.15, 'Strategic Import (9am-3pm)'),
    (peak_export_hours, colors['strategic_export'], 0.15, 'Strategic Export (6pm-10pm)')
]:
    for hour in hour_range:
        period_start = april_day_3.index[0].normalize() + pd.Timedelta(hours=hour)
        period_end = period_start + pd.Timedelta(hours=1)
        ax2.axvspan(period_start, period_end, alpha=alpha, color=color)

ax2.plot(april_day_3.index, april_day_3["AC_Power_kW_osm_total"], 
         label="PV Power", color=colors['pv'], linewidth=1.5)
ax2.plot(april_day_3.index, april_day_3["Home_Load_kW"], 
         label="Load Power", color=colors['load'], linewidth=1.5)

# Battery power Day 3
battery_positive_3 = april_day_3['Battery_Power_kW'].clip(lower=0)
battery_negative_3 = april_day_3['Battery_Power_kW'].clip(upper=0)
ax2.plot(april_day_3.index, battery_positive_3, 
         label="Battery Charging", color=colors['battery_charge'], linewidth=1.5)
ax2.plot(april_day_3.index, battery_negative_3, 
         label="Battery Discharging", color=colors['battery_discharge'], linewidth=1.5)

# Grid power with area fills Day 3
export_power_3 = april_day_3['Grid_Power_KW'].clip(lower=0)
import_power_3 = april_day_3['Grid_Power_KW'].clip(upper=0).abs()
ax2.fill_between(april_day_3.index, export_power_3, alpha=0.4, color=colors['export_fill'])
ax2.fill_between(april_day_3.index, -import_power_3, alpha=0.4, color=colors['import_fill'])
ax2.plot(april_day_3.index, export_power_3, label="Export", color=colors['export'], linewidth=1.7, linestyle='--')
ax2.plot(april_day_3.index, -import_power_3, label="Import", color=colors['import'], linewidth=1.2, linestyle='--')

ax2.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
ax2.set_ylim(power_ylim)

# Configure grids
ax2.grid(True, which='major', linestyle='-', alpha=0.3, linewidth=0.8)
ax2.grid(True, which='minor', linestyle='--', alpha=0.3, linewidth=0.7)

# X-axis formatting Day 3
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=4))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:00'))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center', fontname='Garamond', fontsize=15)

# DAY 1 - BATTERY SOC
ax3.plot(april_day_1.index, april_day_1["Battery_SOC_kWh"], 
         label="Battery SoC", color=colors['soc'], linewidth=2.0)
max_soc_line = ax3.axhline(soc_max, color=colors['soc_limit'], linestyle='--', linewidth=1.5, alpha=0.8)
min_soc_line = ax3.axhline(soc_min, color=colors['soc_limit'], linestyle='--', linewidth=1.5, alpha=0.8)

ax3.set_ylim([0, battery_capacity_kwh * 1.05])
ax3.set_ylabel("SoC [kWh]", fontname='Garamond', fontsize=18, fontweight='bold')
ax3.set_xlabel("Time", fontname='Garamond', fontsize=17, fontweight='bold')

# Add grid lines to SOC graph
ax3.grid(True, which='major', linestyle='-', alpha=0.3, linewidth=0.8)
ax3.grid(True, which='minor', linestyle='--', alpha=0.2, linewidth=0.5)

# X-axis formatting
ax3.xaxis.set_major_locator(mdates.HourLocator(interval=4))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:00'))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0, ha='center', fontname='Garamond', fontsize=15)

# DAY 3 - BATTERY SOC
ax4.plot(april_day_3.index, april_day_3["Battery_SOC_kWh"], 
         label="Battery SoC", color=colors['soc'], linewidth=2.0)
ax4.axhline(soc_max, color=colors['soc_limit'], linestyle='--', linewidth=1.5, alpha=0.8)
ax4.axhline(soc_min, color=colors['soc_limit'], linestyle='--', linewidth=1.5, alpha=0.8)

ax4.set_ylim([0, battery_capacity_kwh * 1.05])
ax4.set_xlabel("Time", fontname='Garamond', fontsize=17, fontweight='bold')

# Add grid lines to SOC graph
ax4.grid(True, which='major', linestyle='-', alpha=0.3, linewidth=0.8)
ax4.grid(True, which='minor', linestyle='--', alpha=0.2, linewidth=0.5)

# X-axis formatting
ax4.xaxis.set_major_locator(mdates.HourLocator(interval=4))
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:00'))
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=0, ha='center', fontname='Garamond', fontsize=15)

# Create custom legend handles including strategic shading and SOC limits
power_handles, power_labels = ax1.get_legend_handles_labels()

# Add strategic shading patches to legend
strategic_import_patch = plt.Rectangle((0,0), 1, 1, fc=colors['strategic_import'], alpha=0.15, 
                                      label='Strategic Import (9am-3pm)')
strategic_export_patch = plt.Rectangle((0,0), 1, 1, fc=colors['strategic_export'], alpha=0.15, 
                                      label='Strategic Export (6pm-10pm)')

# Add SOC limit lines to legend
soc_limit_line = plt.Line2D([0], [0], color=colors['soc_limit'], linestyle='--', linewidth=1.5, 
                           alpha=0.8, label='Max/Min SoC')

# Combine all handles and labels
all_handles = power_handles + [strategic_import_patch, strategic_export_patch, soc_limit_line]
all_labels = power_labels + ['Strategic Import (9am-3pm)', 'Strategic Export (6pm-10pm)', 'Max/Min SoC']

# Single legend for all subplots at the bottom
fig.legend(all_handles, all_labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
           ncol=5, framealpha=0.9, fontsize=12, fancybox=True, shadow=True)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Make space for the legend
plt.savefig('april_day_1_3_arbitrage_separate.pdf', dpi=600, bbox_inches='tight', 
            facecolor='white', edgecolor='none', format='pdf')
plt.show()

print("April 1st and 3rd arbitrage plot generated successfully!")
print("File saved: april_day_1_3_arbitrage_separate.pdf")
print("✓ Day 2 completely removed")
print("✓ 4-column legend with strategic shading colors")
print("✓ Strategic import hours (9am-3pm) - Orange shading")
print("✓ Strategic export hours (6pm-10pm) - Green shading")
print("✓ Max/Min SoC limits added to legend")
print("✓ Rotation=0 for x-axis labels") 
print("✓ Power y-axis limited to -30kW")
print("✓ Grid lines on SOC graphs")
