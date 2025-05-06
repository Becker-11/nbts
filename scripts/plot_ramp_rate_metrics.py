import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 1) locate all your CSVs
project_root = Path(__file__).parent.parent          # nbts/
base = project_root / "experiments" / "temp_time_grid_initial"
print("Looking for CSVs in:", base)
csv_paths = list(base.glob("*/metrics_by_ramp_rate.csv"))
if not csv_paths:
    raise FileNotFoundError(f"No CSVs found under {base}")

# 2) load & annotate each
dfs = []
for p in csv_paths:
    stem = p.parent.name               # e.g. "100C_6h"
    temp_str, time_str = stem.split("_")
    temp_C      = float(temp_str.rstrip("C"))
    hold_time_h = float(time_str.rstrip("h"))
    
    df = pd.read_csv(p)
    df["temp_C"]      = temp_C
    df["hold_time_h"] = hold_time_h
    dfs.append(df)

# 3) combine into one big DataFrame
df_all = pd.concat(dfs, ignore_index=True)

# 4) pick the hold_times to plot
hold_times = sorted(df_all["hold_time_h"].unique())

# 5) metrics to visualize
metrics = ["AUC", "mean_U", "median_U", "d50_nm"]

# 6) prepare output folder
outdir = Path("experiments/ramp_rate_plots")
outdir.mkdir(exist_ok=True)

# 7) loop and save
for ht in hold_times:
    sub = df_all[df_all["hold_time_h"] == ht]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    fig.suptitle(f"Metrics vs. Ramp Rate & Temperature (hold_time = {ht} h)", fontsize=16)
    
    for ax, metric in zip(axes.ravel(), metrics):
        piv = sub.pivot(index="temp_C", columns="ramp_rate", values=metric)
        X, Y = np.meshgrid(piv.columns.values, piv.index.values)
        
        pcm = ax.pcolormesh(X, Y, piv.values, shading="auto")
        fig.colorbar(pcm, ax=ax, label=metric)
        
        ax.set_xlabel("Ramp rate (°C/min)")
        ax.set_ylabel("Bake temp (°C)")
        ax.set_title(metric)
    
    plt.tight_layout(rect=[0,0,1,0.96])
    # save the figure
    fname = outdir / f"ramp_rate_metrics_{int(ht)}h.png"
    fig.savefig(fname, dpi=300)
    plt.close(fig)

print(f"Saved plots to {outdir.resolve()}")
