#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ─── Paths ─────────────────────────────────────────────────────────────────────
project_root = Path(__file__).parent.parent          # nbts/
base_dir = project_root / "experiments" / "temp_time_grid"
heatmap_dir = project_root / "experiments" / "heatmaps"
heatmap_dir.mkdir(exist_ok=True)

# ─── Gather all comparison_stats.csv into one DataFrame ────────────────────────
records = []
for sub in sorted(base_dir.iterdir()):
    stats_file = sub / "comparison_stats.csv"
    if not stats_file.exists():
        continue

    # parse folder name "100C_48h" → (100.0, 48.0)
    temp_str, time_str = sub.name.split("_")
    temp_C      = float(temp_str.rstrip("C"))
    hold_time_h = float(time_str.rstrip("h"))

    df = pd.read_csv(stats_file)
    df["temp_C"]      = temp_C
    df["hold_time_h"] = hold_time_h
    records.append(df)

if not records:
    raise FileNotFoundError(f"No comparison_stats.csv found under {base_dir!s}")

all_stats = pd.concat(records, ignore_index=True)

# ─── Plot all ramp‐rate p‐value heatmaps on one row ───────────────────────────
ramp_rates = sorted(all_stats["ramp_rate"].unique())
n = len(ramp_rates)

fig, axes = plt.subplots(
    1, n,
    figsize=(5 * n, 6),
    sharey=True,
    constrained_layout=True
)

# draw each ramp‐rate’s heatmap
for ax, rr in zip(axes, ramp_rates):
    sub = all_stats[all_stats["ramp_rate"] == rr]
    piv = sub.pivot(index="temp_C", columns="hold_time_h", values="p_value")
    X, Y = np.meshgrid(piv.columns.values, piv.index.values)
    Z = piv.values

    pcm = ax.pcolormesh(
        X, Y, Z,
        shading="auto",
        cmap="cividis",
        vmin=0, vmax=1
    )
    ax.set_title(f"{rr} °C/min vs 1 °C/min")
    ax.set_xlabel("Hold time (h)")
    if ax is axes[0]:
        ax.set_ylabel("Bake temperature (°C)")

# add one shared colorbar on the right
cbar = fig.colorbar(
    pcm,
    ax=axes,
    orientation="vertical",
    fraction=0.02,
    pad=0.04
)
cbar.set_label("p-value")

# overlay red contour where p < 0.05
for ax, rr in zip(axes, ramp_rates):
    sub = all_stats[all_stats["ramp_rate"] == rr]
    piv = sub.pivot(index="temp_C", columns="hold_time_h", values="p_value")
    X, Y = np.meshgrid(piv.columns.values, piv.index.values)
    mask = piv.values < 0.05

    cs = ax.contour(
        X, Y, mask.astype(float),
        levels=[0.5],
        colors="red",
        linewidths=2
    )
    ax.clabel(cs, fmt={0.5: "p<0.05"}, inline=True, fontsize=8)

fig.suptitle("p-value heatmaps for all ramp rates", fontsize=16)

out_file = heatmap_dir / "pvalue_heatmaps_combined.png"
fig.savefig(out_file, dpi=300)
plt.close(fig)



print(f"Heatmaps saved to {heatmap_dir.resolve()}")
