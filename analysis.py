import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Base directory containing simulation folders (named like "sim_t0.1_T100", etc.)
base_dir = 'sim_output'
subfolders = glob.glob(os.path.join(base_dir, 'sim_t*_T*'))

# Lists to store simulation parameters and metrics
time_vals = []
temp_vals = []
ratio_vals = []       # Metric A: (max current density)/(surface current density)
x_peak_vals = []      # Metric B: x-position where current density peaks
surf_ratio_vals = []  # Metric C: (current density at surface)/(J_max at surface)

for folder in subfolders:
    folder_name = os.path.basename(folder)  # e.g., "sim_t0.1_T100"
    parts = folder_name.split('_')           # e.g., ["sim", "t0.1", "T100"]
    
    # Parse time and temperature from folder name
    try:
        time_val = float(parts[1][1:])  # remove the leading 't'
        temp_val = float(parts[2][1:])  # remove the leading 'T'
    except (IndexError, ValueError):
        continue
    
    # Path to the CSV file containing current-density data
    csv_path = os.path.join(folder, 'current_density.csv')
    if not os.path.exists(csv_path):
        continue
    
    # Read the CSV; skip initial spaces so that column names are clean.
    df = pd.read_csv(csv_path, skipinitialspace=True)
    # Expect columns: "x", "current_density", "J_max", "J_min"
    if not {"x", "current_density", "J_min"}.issubset(df.columns):
        continue

    # -----------------------------
    # Metric A: (max current density) / (current density at surface)
    J_max_val = df["current_density"].max()
    # Try to get the row where x == 0. If not found, use the row with the minimum x value.
    surface_df = df.loc[df["x"] == 0, "J_min"]
    if surface_df.empty:
        surface_df = df.loc[[df["x"].idxmin()], "J_min"]
    J_surface_val = surface_df.iloc[0]
    ratio_A = J_max_val / J_surface_val

    # -----------------------------
    # Metric B: x position where the current density peaks
    x_peak = df.loc[df["current_density"].idxmax(), "x"]

    # -----------------------------
    # Metric C: (current density at surface) / (J_max at surface)
    surface_Jmax_df = df.loc[df["x"] == 0, "current_density"]
    if surface_Jmax_df.empty:
        surface_Jmax_df = df.loc[[df["x"].idxmin()], "curent_density"]
    J_max_surface = surface_Jmax_df.iloc[0]
    ratio_C = J_max_surface / J_surface_val

    # Save simulation parameters and computed metrics
    time_vals.append(time_val)
    temp_vals.append(temp_val)
    ratio_vals.append(ratio_A)
    x_peak_vals.append(x_peak)
    surf_ratio_vals.append(ratio_C)

# Convert lists to NumPy arrays
time_vals = np.array(time_vals)
temp_vals = np.array(temp_vals)
ratio_vals = np.array(ratio_vals)
x_peak_vals = np.array(x_peak_vals)
surf_ratio_vals = np.array(surf_ratio_vals)

# Create a grid for the unique simulation times and temperatures
unique_times = np.unique(time_vals)
unique_temps = np.unique(temp_vals)

# Initialize 2D arrays (rows: temperature, columns: time) for each metric
Z_ratio = np.full((len(unique_temps), len(unique_times)), np.nan)       # Metric A
Z_xpeak = np.full((len(unique_temps), len(unique_times)), np.nan)       # Metric B
Z_surf_ratio = np.full((len(unique_temps), len(unique_times)), np.nan)  # Metric C

# Fill the 2D arrays using the simulation parameters as indices
for (t, T, r, xp, sr) in zip(time_vals, temp_vals, ratio_vals, x_peak_vals, surf_ratio_vals):
    i_time = np.where(unique_times == t)[0][0]
    i_temp = np.where(unique_temps == T)[0][0]
    Z_ratio[i_temp, i_time] = r
    Z_xpeak[i_temp, i_time] = xp
    Z_surf_ratio[i_temp, i_time] = sr

# Create a meshgrid for plotting
X, Y = np.meshgrid(unique_times, unique_temps)

# Create the "analysis" folder if it doesn't exist
analysis_dir = "analysis"
if not os.path.exists(analysis_dir):
    os.makedirs(analysis_dir)


# -----------------------------
# Plot 1: Metric A – Original Ratio
plt.figure(figsize=(8, 6))
mesh1 = plt.pcolormesh(X, Y, Z_ratio, shading="gouraud", cmap='cividis')
cbar = plt.colorbar(mesh1, label='maximum J(x) / maximum supercurrent in clean Nb')
cbar.set_label('maximum J(x) / maximum supercurrent in clean Nb', fontsize=12)
cs1 = plt.contour(X, Y, Z_ratio, levels=10, colors='white', linewidths=1)
plt.clabel(cs1, inline=True, fontsize=8)
plt.xlabel('Time (h)', fontsize=14)
plt.ylabel('Temperature (°C)', fontsize=14)
#plt.title('Max Current Density over Surface Current Density')
plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, 'ratio_max_over_surface.pdf'))
plt.close()

# -----------------------------
# Plot 2: Metric B – x Position of the Current Density Peak
plt.figure(figsize=(8, 6))
mesh2 = plt.pcolormesh(X, Y, Z_xpeak, shading="gouraud", cmap='cividis')
cbar1 = plt.colorbar(mesh2, label='x position of supercurrent density peak')
cbar1.set_label('maximum J(x) / maximum supercurrent in clean Nb', fontsize=14)  # Adjust the label font size
cs2 = plt.contour(X, Y, Z_xpeak, levels=3, colors='white', linewidths=1)
plt.clabel(cs2, inline=True, fontsize=20)
plt.xlabel('Time (h)', fontsize=16)
plt.ylabel('Temperature (°C)', fontsize=16)
#plt.title('x Position of Current Density Peak')
plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, 'x_peak_position.pdf'))
plt.close()

# -----------------------------
# Plot 3: Metric C – Surface Current Density Ratio
plt.figure(figsize=(8, 6))
mesh3 = plt.pcolormesh(X, Y, Z_surf_ratio, shading="gouraud", cmap='cividis')
cbar2 = plt.colorbar(mesh3, label='J(x) surface value / maximum supercurrent in clean Nb')
cbar2.set_label('maximum J(x) / maximum supercurrent in clean Nb', fontsize=14)
cs3 = plt.contour(X, Y, Z_surf_ratio, levels=10, colors='white', linewidths=1)
plt.clabel(cs3, inline=True, fontsize=12)
plt.xlabel('Time (h)', fontsize=16)
plt.ylabel('Temperature (°C)', fontsize=16)
#plt.title('Surface Current Density Ratio')
plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, 'surface_current_ratio.pdf'))
plt.close()
