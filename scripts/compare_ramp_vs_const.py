#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ─── PARAMETERS ────────────────────────────────────────────────────────────────
BASE_DIR      = Path("experiments") / "temp_time_grid"
OUT_DIR       = Path("experiments") / "heatmaps"
OUT_DIR.mkdir(exist_ok=True)
MET_CONST     = "metrics_const.csv"
MET_RAMP      = "metrics_by_ramp_rate.csv"
METRICS       = ["AUC", "mean_U", "median_U", "d50_nm"]  # order for panels

# ─── COLLECT ABSOLUTE Δ‐METRICS ────────────────────────────────────────────────
records = []
for sub in sorted(BASE_DIR.iterdir()):
    const_fp = sub / MET_CONST
    ramp_fp  = sub / MET_RAMP
    if not (const_fp.exists() and ramp_fp.exists()):
        continue

    # parse folder name "100C_48h"
    T_str, t_str = sub.name.split("_")
    temp_C      = float(T_str.rstrip("C"))
    hold_h      = float(t_str.rstrip("h"))

    df_const = pd.read_csv(const_fp)
    df_ramp  = pd.read_csv(ramp_fp)

    const_vals = df_const.iloc[0]

    for _, row in df_ramp.iterrows():
        rr = row["ramp_rate"]
        for metric in METRICS:
            # absolute difference so largest effect → largest positive
            delta = abs(row[metric] - const_vals[metric])
            records.append({
                "temp_C":      temp_C,
                "hold_time_h": hold_h,
                "ramp_rate":   rr,
                "metric":      metric,
                "delta":       delta
            })

df = pd.DataFrame(records)
if df.empty:
    raise RuntimeError("No data found! Check that CSVs exist under each subfolder.")

# ─── PLOT 2×2 PANELS FOR EACH RAMP RATE ────────────────────────────────────────
ramp_rates = sorted(df["ramp_rate"].unique())

for rr in ramp_rates:
    sub = df[df["ramp_rate"] == rr]

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 10),
        sharex=True, sharey=True,
        constrained_layout=True
    )
    axes = axes.ravel()

    for ax, metric in zip(axes, METRICS):
        piv = (
            sub[sub["metric"] == metric]
            .pivot(index="temp_C", columns="hold_time_h", values="delta")
        )

        X, Y = np.meshgrid(piv.columns.values, piv.index.values)
        Z    = piv.values

        pcm = ax.pcolormesh(
            X, Y, Z,
            shading="auto",
            cmap="cividis",
            vmin=0, vmax=np.nanmax(Z)
        )

        ax.set_title(metric)
        ax.set_xlabel("Hold time (h)")
        ax.set_ylabel("Bake temperature (°C)")
        fig.colorbar(pcm, ax=ax, label=f"|Δ {metric}|")

    fig.suptitle(f"Absolute Δ‐metrics (ramp {rr} °C/min vs constant T)", fontsize=16)
    out_file = OUT_DIR / f"abs_delta_metrics_ramp_{int(rr)}.png"
    fig.savefig(out_file, dpi=300)
    plt.close(fig)

print("Done! Absolute‐difference heatmaps saved to:", OUT_DIR.resolve())
