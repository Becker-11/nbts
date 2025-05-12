import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── USER CONFIG ────────────────────────────────────────────────────────────────
BASE_DIR   = "experiments/ramp_rate_comparison"
PLOTS_DIR  = "experiments/ramp_rate_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

RAMP_RATES = [1, 2, 5, 10]              # °C/min
CONST_FILE = "oxygen_profile_const.csv"

OUT_MAX_PDF = os.path.join(PLOTS_DIR, "max_diff_heatmaps.pdf")
OUT_DEP_PDF = os.path.join(PLOTS_DIR, "depth_at_max_heatmaps.pdf")
# ────────────────────────────────────────────────────────────────────────────────

# 1) discover all subfolders like "100C_6h"
all_dirs = [
    d for d in glob.glob(os.path.join(BASE_DIR, "*"))
    if os.path.isdir(d) and d.endswith("h") and "C_" in os.path.basename(d)
]

# 2) parse out all unique temps & hours
temps, hours = [], []
for d in all_dirs:
    name    = os.path.basename(d)         # e.g. "120C_42h"
    core    = name[:-1]                   # drop trailing 'h' → "120C_42"
    temp_s, hr_s = core.split("C_")       # → ["120", "42"]
    try:
        temps.append(float(temp_s))
        hours.append(float(hr_s))
    except ValueError:
        continue

unique_temps = sorted(set(temps))
unique_hours = sorted(set(hours))

# 3) prepare empty 2D arrays for each ramp rate
Z_max = {r: np.full((len(unique_temps), len(unique_hours)), np.nan)
         for r in RAMP_RATES}
Z_dep = {r: np.full((len(unique_temps), len(unique_hours)), np.nan)
         for r in RAMP_RATES}

# 4) loop, compute metrics
for d in all_dirs:
    name    = os.path.basename(d)
    core    = name[:-1]
    temp_s, hr_s = core.split("C_")
    try:
        t = float(temp_s)
        h = float(hr_s)
    except ValueError:
        continue

    i_t = unique_temps .index(t)
    i_h = unique_hours.index(h)

    # load constant profile
    const_fp = os.path.join(d, CONST_FILE)
    if not os.path.exists(const_fp):
        print(f"⚠️ missing constant in {d}, skipping")
        continue

    df0 = pd.read_csv(const_fp)
    x  = df0.iloc[:, 0].to_numpy()   # depth
    c0 = df0.iloc[:, -1].to_numpy()  # concentration

    # for each ramp rate
    for r in RAMP_RATES:
        fp = os.path.join(d, f"oxygen_profile_{r}C_min.csv")
        if not os.path.exists(fp):
            continue

        df  = pd.read_csv(fp)
        cr  = df.iloc[:, -1].to_numpy()
        diff = np.abs(cr - c0)
        idx  = np.nanargmax(diff)

        Z_max[r][i_t, i_h] = diff[idx]
        Z_dep[r][i_t, i_h] = x[idx]

        print(f"Folder {name}, {r}°C/min → "
              f"max|ΔC|={diff[idx]:.2e} at depth={x[idx]:.2e}")

# 5) build meshgrid for plotting
H, T = np.meshgrid(unique_hours, unique_temps)

# 6) Plot A: max|ΔC|
fig1, axes1 = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axes1 = axes1.flatten()
for ax, r in zip(axes1, RAMP_RATES):
    pcm = ax.pcolormesh(H, T, Z_max[r], shading="auto")
    cb  = fig1.colorbar(pcm, ax=ax, label="Max |ΔC|")
    cs  = ax.contour(H, T, Z_max[r], levels=8, colors="white", linewidths=1)
    ax.clabel(cs, inline=True, fmt="%.2e", fontsize=8)
    ax.set_title(f"{r} °C/min")
    ax.set_xlabel("Hold time (h)")
    ax.set_ylabel("Bake temp (°C)")

fig1.suptitle("Maximum difference vs. constant profile", y=0.95)
fig1.tight_layout()
fig1.savefig(OUT_MAX_PDF)
print(f"Wrote {OUT_MAX_PDF}")

# 7) Plot B: depth at max|ΔC|
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axes2 = axes2.flatten()
for ax, r in zip(axes2, RAMP_RATES):
    pcm = ax.pcolormesh(H, T, Z_dep[r], shading="auto")
    cb  = fig2.colorbar(pcm, ax=ax, label="Depth of max ΔC")
    cs  = ax.contour(H, T, Z_dep[r], levels=8, colors="white", linewidths=1)
    ax.clabel(cs, inline=True, fmt="%.2f", fontsize=8)
    ax.set_title(f"{r} °C/min")
    ax.set_xlabel("Hold time (h)")
    ax.set_ylabel("Bake temp (°C)")

fig2.suptitle("Depth at which maximum difference occurs", y=0.95)
fig2.tight_layout()
fig2.savefig(OUT_DEP_PDF)
print(f"Wrote {OUT_DEP_PDF}")
