#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
from scipy.stats import linregress, ks_2samp

from config.sim_config import load_sim_config
from simulation.temp_profile import TimeDepProfile
from simulation.cn_solver    import CNSolver
from simulation.ciovati_model import CiovatiModel

# ─── Experiment parameters ─────────────────────────────────────────────────────
config_path    = "config/sim_config.yml"
bake_temp_C    = 120.0                   # °C
hold_time_h    = 48.0                     # hours
ramp_rates     = [1, 2, 5, 10]      # °C/min

base_exp_dir   = "experiments"
run_subdir     = f"{int(bake_temp_C)}C_{int(hold_time_h)}h"
exp_dir        = os.path.join(base_exp_dir, run_subdir)
os.makedirs(exp_dir, exist_ok=True)
_paths = lambda fn: os.path.join(exp_dir, fn)

# filenames
plot_cmp     = "ramp_rate_comparison.pdf"
plot_contour = "concentration_contour.pdf"
plot_roi     = "roi_metrics_vs_ramp.pdf"
plot_diff    = "difference_profiles.pdf"
metrics_csv  = "metrics_by_ramp_rate.csv"
trend_csv    = "linear_trend_summary.csv"
func_csv     = "functional_permutation_summary.csv"
stats_csv    = "comparison_stats.csv"

# ROI & zoom
roi_depths_nm = [5.0, 50.0]
pen_depth_nm  = 100.0
zoom_depth_nm = 150.0
zoom_depth_nm_contour = 100.0
# ───────────────────────────────────────────────────────────────────────────────

# load config & prepare
base_cfg = load_sim_config(config_path)
profiles, metrics, comparison_stats = [], [], []

all_x = None

# ─── 1) Generate curves & metrics ──────────────────────────────────────────────
fig1, (ax_o, ax_t) = plt.subplots(1, 2, figsize=(12, 5))
ref_profile = None
for rr in ramp_rates:
    cfg = deepcopy(base_cfg)
    cfg.temp_profile.ramp_rate_C_per_min = rr

    # temperature profile
    start_K = cfg.temp_profile.start_C + 273.15
    bake_K  = bake_temp_C + 273.15
    t_h, T_K, _ = TimeDepProfile(cfg, start_K, bake_K, hold_time_h).generate()

    # oxygen profile
    solver   = CNSolver(cfg, T_K, hold_time_h, CiovatiModel(cfg.ciovati))
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

# decorate & shared legend
ax_o.set_xlim(0, zoom_depth_nm)
ax_o.set_xlabel("Depth (nm)"); ax_o.set_ylabel("Oxygen concentration")
ax_o.set_title(f"O₂ @ {int(bake_temp_C)}°C, {int(hold_time_h)}h")
ax_t.set_xlabel("Time (h)"); ax_t.set_ylabel("Temperature (°C)")
ax_t.set_title("T vs time")
h_o, l_o = ax_o.get_legend_handles_labels()
fig1.legend(h_o, l_o, loc="center right", title="Ramp rate", frameon=True)
fig1.tight_layout(rect=[0, 0, 0.85, 1])
fig1.savefig(_paths(plot_cmp), dpi=300)

# ─── 2) Contour plot ───────────────────────────────────────────────────────────
profile_matrix = np.vstack(profiles)
fig2, ax2 = plt.subplots(figsize=(6, 5))
cf = ax2.contourf(all_x, ramp_rates, profile_matrix, levels=20, cmap="viridis")
fig2.colorbar(cf, ax=ax2, label="O concentration")
ax2.set_xlim(0, zoom_depth_nm_contour); ax2.set_xlabel("Depth (nm)")
ax2.set_ylabel("Ramp rate (°C/min)")
ax2.set_title("Conc vs depth & ramp rate")
fig2.tight_layout(); fig2.savefig(_paths(plot_contour), dpi=300)

# ─── 3) ROI metrics plot ───────────────────────────────────────────────────────
roi_idxs = [np.argmin(np.abs(all_x - d)) for d in roi_depths_nm]
pen_idx  = np.argmin(np.abs(all_x - pen_depth_nm))
fig3, ax3 = plt.subplots(figsize=(6, 5))
for d, i in zip(roi_depths_nm, roi_idxs):
    ax3.plot(ramp_rates, profile_matrix[:, i], '-o', label=f"U@{d:.0f}nm")
auc_tot  = np.trapz(profile_matrix, all_x, axis=1)
auc_up   = np.trapz(profile_matrix[:, :pen_idx+1], all_x[:pen_idx+1], axis=1)
ax3.plot(ramp_rates, auc_up/auc_tot, '-s', label=f"Frac ≤{pen_depth_nm:.0f}nm")
ax3.set_xlabel("Ramp rate (°C/min)"); ax3.set_ylabel("Metric")
ax3.set_title("ROI metrics"); ax3.legend()
fig3.tight_layout(); fig3.savefig(_paths(plot_roi), dpi=300)

# ─── 4) Difference plot & statistical comparison ───────────────────────────────
fig4, (ax4, ax5) = plt.subplots(1, 2, figsize=(12, 5))
for rr, U in zip(ramp_rates, profiles):
    ax4.plot(all_x, U - ref_profile, label=f"{rr:.2g} °C/min")
    # KS test vs reference (1 °C/min)
    ks_stat, p_value = ks_2samp(ref_profile, U)
    # Euclidean distance
    euclidean_dist = np.sqrt(np.sum((ref_profile - U) ** 2))
    comparison_stats.append({
        "ramp_rate": rr,
        "ks_stat": ks_stat,
        "p_value": p_value,
        "euclidean_dist": euclidean_dist
    })

ax4.axhline(0, ls='--', color='k'); ax4.set_xlim(0, zoom_depth_nm)
ax4.set_xlabel("Depth (nm)"); ax4.set_ylabel("Δ[O] rel 1°C/min")
ax4.set_title("Difference between profiles"); ax4.legend()

# Plot summary statistics
ax5.boxplot([p for p in profiles], tick_labels=ramp_rates)
ax5.set_xlabel("Ramp rate (°C/min)"); ax5.set_ylabel("Oxygen concentration")
ax5.set_title("Distribution Summary")
fig4.tight_layout(); fig4.savefig(_paths(plot_diff), dpi=300)


# Save comparison stats
dfs = pd.DataFrame(comparison_stats)
dfs.to_csv(_paths(stats_csv), index=False)