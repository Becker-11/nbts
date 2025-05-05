#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
from scipy.stats import ks_2samp

from config.sim_config import load_sim_config
from simulation.temp_profile import TimeDepProfile
from simulation.cn_solver    import CNSolver
from simulation.ciovati_model import CiovatiModel

# ─── Experiment parameters ─────────────────────────────────────────────────────
config_path    = "config/sim_config.yml"
bake_temps_C   = np.arange(100, 601, 20)  # e.g., 100°C to 600°C in 100°C steps
hold_times_h   = np.arange(6, 97, 6)     # e.g., 12h to 48h in 12h steps
ramp_rates     = [1, 2, 5, 10]             # °C/min

base_exp_dir   = "experiments"
exp_dir        = os.path.join(base_exp_dir, "temp_time_grid")
os.makedirs(exp_dir, exist_ok=True)
_paths = lambda fn: os.path.join(exp_dir, fn)

# filenames
plot_cmp_base  = "ramp_rate_comparison_{temp}C_{time}h.pdf"
plot_diff_base = "difference_profiles_{temp}C_{time}h.pdf"
metrics_csv    = "metrics_by_ramp_rate.csv"
stats_csv      = "comparison_stats_grid.csv"
heatmap_pval   = "heatmap_p_values.png"

# ROI & zoom
roi_depths_nm  = [5.0, 50.0]
pen_depth_nm   = 100.0
zoom_depth_nm  = 150.0
zoom_depth_nm_contour = 100.0

# Initialize storage for grid results
p_values_grid = np.zeros((len(bake_temps_C), len(hold_times_h), len(ramp_rates) - 1))  # Exclude ref (1 °C/min)

# Load base config
base_cfg = load_sim_config(config_path)

# ─── 1) Generate curves & metrics over grid ────────────────────────────────────
for i, temp in enumerate(bake_temps_C):
    for j, time in enumerate(hold_times_h):
        run_subdir = f"{int(temp)}C_{int(time)}h"
        exp_run_dir = os.path.join(exp_dir, run_subdir)
        os.makedirs(exp_run_dir, exist_ok=True)
        _paths_run = lambda fn: os.path.join(exp_run_dir, fn)

        profiles, metrics, comparison_stats = [], [], []
        all_x = None
        ref_profile = None

        fig1, (ax_o, ax_t) = plt.subplots(1, 2, figsize=(12, 5))
        for rr in ramp_rates:
            cfg = deepcopy(base_cfg)
            cfg.temp_profile.ramp_rate_C_per_min = rr

            # temperature profile
            start_K = cfg.temp_profile.start_C + 273.15
            bake_K  = temp + 273.15
            t_h, T_K, _ = TimeDepProfile(cfg, start_K, bake_K, time).generate()

            # oxygen profile
            print(f"Running {run_subdir} @ {rr:.2g} °C/min")
            solver   = CNSolver(cfg, T_K, time, CiovatiModel(cfg.ciovati))
            U_final  = solver.get_oxygen_profile()[-1]
            x_grid   = np.linspace(0, cfg.grid.x_max_nm, cfg.grid.n_x)

            profiles.append(U_final)
            all_x = x_grid

            # scalar summaries
            auc   = np.trapz(U_final, x_grid)
            meanU = U_final.mean()
            stdU  = U_final.std()
            medU  = np.median(U_final)
            half  = U_final[0]/2
            idx   = np.where(U_final<=half)[0]
            d50   = x_grid[idx[0]] if idx.size else np.nan
            metrics.append({
                "temp_C": temp,
                "hold_time_h": time,
                "ramp_rate": rr,
                "AUC":       auc,
                "mean_U":    meanU,
                "std_U":     stdU,
                "median_U":  medU,
                "d50_nm":    d50
            })

            # plotting
            ax_o.plot(x_grid, U_final, label=f"{rr:.2g} °C/min")
            ax_t.plot(t_h, T_K-273.15)
            if rr == 1:
                ref_profile = U_final

        # decorate & save comparison plot
        ax_o.set_xlim(0, zoom_depth_nm)
        ax_o.set_xlabel("Depth (nm)"); ax_o.set_ylabel("Oxygen concentration")
        ax_o.set_title(f"O₂ @ {int(temp)}°C, {int(time)}h")
        ax_t.set_xlabel("Time (h)"); ax_t.set_ylabel("Temperature (°C)")
        ax_t.set_title("T vs time")
        h_o, l_o = ax_o.get_legend_handles_labels()
        fig1.legend(h_o, l_o, loc="center right", title="Ramp rate", frameon=True)
        fig1.tight_layout(rect=[0, 0, 0.85, 1])
        fig1.savefig(_paths_run(plot_cmp_base.format(temp=int(temp), time=int(time))), dpi=300)
        plt.close(fig1)

        # Difference plot & statistical comparison
        fig4, (ax4, ax5) = plt.subplots(1, 2, figsize=(12, 5))
        for k, (rr, U) in enumerate(zip(ramp_rates[1:], profiles[1:])):
            ax4.plot(all_x, U - ref_profile, label=f"{rr:.2g} °C/min")
            ks_stat, p_value = ks_2samp(ref_profile, U)
            euclidean_dist = np.sqrt(np.sum((ref_profile - U) ** 2))
            comparison_stats.append({
                "temp_C": temp,
                "hold_time_h": time,
                "ramp_rate": rr,
                "ks_stat": ks_stat,
                "p_value": p_value,
                "euclidean_dist": euclidean_dist
            })
            p_values_grid[i, j, k] = p_value

        ax4.axhline(0, ls='--', color='k'); ax4.set_xlim(0, zoom_depth_nm)
        ax4.set_xlabel("Depth (nm)"); ax4.set_ylabel("Δ[O] rel 1°C/min")
        ax4.set_title("Difference between profiles"); ax4.legend()
        ax5.boxplot([p for p in profiles], tick_labels=ramp_rates)
        ax5.set_xlabel("Ramp rate (°C/min)"); ax5.set_ylabel("Oxygen concentration")
        ax5.set_title("Distribution Summary")
        fig4.tight_layout()
        fig4.savefig(_paths_run(plot_diff_base.format(temp=int(temp), time=int(time))), dpi=300)
        plt.close(fig4)

        # Save run-specific metrics
        dfm = pd.DataFrame(metrics)
        dfm.to_csv(_paths_run(metrics_csv), index=False)

# ─── 2) Save grid statistics ───────────────────────────────────────────────────
dfs = pd.DataFrame(comparison_stats)
dfs.to_csv(_paths(stats_csv), index=False)

# ─── 3) Generate heatmaps ──────────────────────────────────────────────────────
fig_heat, axes = plt.subplots(1, len(ramp_rates) - 1, figsize=(15, 5), sharey=True)
for k, rr in enumerate(ramp_rates[1:]):
    im = axes[k].imshow(p_values_grid[:, :, k].T, aspect='auto', cmap='cividis',
                       extent=[bake_temps_C[0], bake_temps_C[-1], hold_times_h[0], hold_times_h[-1]],
                       vmin=0, vmax=1)
    axes[k].set_title(f"Ramp {rr} °C/min vs 1 °C/min")
    axes[k].set_xlabel("Temperature (°C)")
    if k == 0:
        axes[k].set_ylabel("Hold Time (h)")
    plt.colorbar(im, ax=axes[k], label="p-value")
    # Overlay significance threshold
    axes[k].contour(bake_temps_C, hold_times_h, p_values_grid[:, :, k].T < 0.05, levels=[0], colors='r')

fig_heat.tight_layout()
fig_heat.savefig(_paths(heatmap_pval), dpi=300)
plt.close(fig_heat)

print("Heatmaps and data saved. Check experiments/temp_time_grid for results.")