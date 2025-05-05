#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ─── Setup ───────────────────────────────────────────────────────────────
base_exp_dir = "experiments"
exp_dir      = os.path.join(base_exp_dir, "temp_time_grid")
_paths       = lambda fn: os.path.join(exp_dir, fn)

# Expected ranges (must match the original script)
bake_temps_C = np.arange(100, 601, 100)  # 100°C to 600°C in 100°C steps
hold_times_h = np.arange(12, 49, 12)     # 12h to 48h in 12h steps
ramp_rates   = [1, 2, 5, 10]             # °C/min

# Filenames
stats_csv    = "comparison_stats_grid.csv"
heatmap_pval = "heatmap_p_values_new_axes.png"

# Ensure directory exists
if not os.path.exists(exp_dir):
    raise FileNotFoundError(f"Directory {exp_dir} does not exist. Run the main script first.")

# ─── Load and Process Data ───────────────────────────────────────────────
# Load the comparison stats
dfs = pd.read_csv(_paths(stats_csv))

# Verify expected temperatures and times
unique_temps = sorted(dfs["temp_C"].unique())
unique_times = sorted(dfs["hold_time_h"].unique())
unique_ramps = sorted(dfs["ramp_rate"].unique())

# Check if data matches expected grid
if not np.allclose(unique_temps, bake_temps_C) or not np.allclose(unique_times, hold_times_h):
    raise ValueError("Temperature or hold time values in CSV do not match expected ranges.")
if not np.allclose(unique_ramps, ramp_rates[1:]):
    raise ValueError("Ramp rates in CSV do not match expected values.")

# Initialize grid for p-values (temp × time × ramp rates excluding 1 °C/min)
p_values_grid = np.zeros((len(bake_temps_C), len(hold_times_h), len(ramp_rates) - 1))

# Populate the grid
for i, temp in enumerate(bake_temps_C):
    for j, time in enumerate(hold_times_h):
        for k, rr in enumerate(ramp_rates[1:]):  # Skip 1 °C/min (reference)
            row = dfs[(dfs["temp_C"] == temp) & (dfs["hold_time_h"] == time) & (dfs["ramp_rate"] == rr)]
            if len(row) == 1:
                p_values_grid[i, j, k] = row["p_value"].iloc[0]
            else:
                print(f"Warning: Missing or duplicate data for {temp}°C, {time}h, {rr}°C/min. Setting p-value to 1.")
                p_values_grid[i, j, k] = 1.0

# ─── Generate Heatmaps ───────────────────────────────────────────────────
fig_heat, axes = plt.subplots(1, len(ramp_rates) - 1, figsize=(15, 5), sharey=True)
for k, rr in enumerate(ramp_rates[1:]):
    # Plot heatmap with hours on x-axis and temperature on y-axis
    # Transpose the grid slice to swap axes
    im = axes[k].imshow(p_values_grid[:, :, k], aspect='auto', cmap='cividis',
                        extent=[hold_times_h[0], hold_times_h[-1], bake_temps_C[-1], bake_temps_C[0]],
                        vmin=0, vmax=1)
    axes[k].set_title(f"Ramp {rr} °C/min vs 1 °C/min")
    axes[k].set_xlabel("Hold Time (h)")
    if k == 0:
        axes[k].set_ylabel("Temperature (°C)")
    plt.colorbar(im, ax=axes[k], label="p-value")
    # Overlay significance threshold
    axes[k].contour(hold_times_h, bake_temps_C[::-1], p_values_grid[:, :, k] < 0.05, levels=[0], colors='r')

fig_heat.tight_layout()
fig_heat.savefig(_paths(heatmap_pval), dpi=300)
plt.close(fig_heat)

print(f"Heatmaps saved to {_paths(heatmap_pval)}")