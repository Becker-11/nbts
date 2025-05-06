#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.stats import ks_2samp

from config.sim_config            import load_sim_config
from simulation.temp_profile      import RampHoldProfile, ConstantProfile
from simulation.cn_solver         import CNSolver
from simulation.ciovati_model     import CiovatiModel

# ─── Experiment parameters ─────────────────────────────────────────────────────
config_path  = "config/sim_config.yml"
bake_temps_C = np.arange(100, 221, 20)   # 100,120,…,300 °C
hold_times_h = np.arange(6, 73, 6)       # 6,12,…,96 h
ramp_rates   = [1]             # °C/min (can be any list)

# ─── Paths & filenames ─────────────────────────────────────────────────────────
exp_dir           = os.path.join("experiments", "temp_time_grid")
os.makedirs(exp_dir, exist_ok=True)
_paths            = lambda fn, subdir="": os.path.join(exp_dir, subdir, fn)

metrics_const_csv = "metrics_const.csv"
metrics_ramp_csv  = "metrics_by_ramp_rate.csv"
stats_csv         = "comparison_stats.csv"

plot_cmp_base     = "ramp_vs_const_{temp}C_{hold}h.pdf"
plot_diff_base    = "diff_vs_const_{temp}C_{hold}h.pdf"

zoom_depth_nm     = 150.0

# ─── Load base config ─────────────────────────────────────────────────────────
base_cfg = load_sim_config(config_path)

for temp in bake_temps_C:
    for hold in hold_times_h:
        subdir = f"{int(temp)}C_{int(hold)}h"
        run_dir = os.path.join(exp_dir, subdir)
        os.makedirs(run_dir, exist_ok=True)

        # ——— 1) CONSTANT profile ———————————————————————————————
        cfg_c = deepcopy(base_cfg)
        start_K = cfg_c.temp_profile.start_C + 273.15
        bake_K  = temp + 273.15

        # generate constant‐T profile
        t_const_h, T_K_const, _ = ConstantProfile(
            cfg_c, start_K, bake_K, hold
        ).generate()

        # solve diffusion
        print(f"Running constant profile @ {temp}°C, {hold}h")
        sol_c = CNSolver(cfg_c, T_K_const, hold, CiovatiModel(cfg_c.ciovati))
        U_const = sol_c.get_oxygen_profile()[-1]

        # x‐grid
        x_grid = np.linspace(0, cfg_c.grid.x_max_nm, cfg_c.grid.n_x)

        # compute scalars
        half = U_const[0]/2
        idx  = np.where(U_const <= half)[0]
        const_metrics = {
            "temp_C":      temp,
            "hold_time_h": hold,
            "AUC":         np.trapz(U_const, x_grid),
            "mean_U":      U_const.mean(),
            "std_U":       U_const.std(),
            "median_U":    np.median(U_const),
            "d50_nm":      (x_grid[idx[0]] if idx.size else np.nan)
        }
        # save
        pd.DataFrame([const_metrics]).to_csv(
            _paths(metrics_const_csv, subdir), index=False
        )

        # containers for ramp results
        ramp_metrics = []
        stats_records = []
        profiles = []

        # ——— 2) RAMP‐HOLD profiles ——————————————————————————————
        for rr in ramp_rates:
            cfg_r = deepcopy(base_cfg)
            cfg_r.temp_profile.ramp_rate_C_per_min = rr

            # generate ramp‐hold T(t)
            t_rh_h, T_K_rh, _ = RampHoldProfile(
                cfg_r, start_K, bake_K, hold
            ).generate()

            print(f"Running ramp profile @ {temp}°C, {hold}h, {rr}°C/min")
            sol_r = CNSolver(cfg_r, T_K_rh, hold, CiovatiModel(cfg_r.ciovati))
            U_r = sol_r.get_oxygen_profile()[-1]
            profiles.append(U_r)

            # ramp metrics
            half_r = U_r[0]/2
            idx_r  = np.where(U_r <= half_r)[0]
            ramp_metrics.append({
                "temp_C":      temp,
                "hold_time_h": hold,
                "ramp_rate":   rr,
                "AUC":         np.trapz(U_r, x_grid),
                "mean_U":      U_r.mean(),
                "std_U":       U_r.std(),
                "median_U":    np.median(U_r),
                "d50_nm":      (x_grid[idx_r[0]] if idx_r.size else np.nan)
            })

            # comparison stats vs constant
            diff = U_r - U_const
            ks_stat, p_val = ks_2samp(U_const, U_r)
            stats_records.append({
                "temp_C":      temp,
                "hold_time_h": hold,
                "ramp_rate":   rr,
                "ks_stat":     ks_stat,
                "p_value":     p_val,
                "euclidean_dist": np.linalg.norm(diff)
            })

        # save ramp CSVs
        pd.DataFrame(ramp_metrics).to_csv(
            _paths(metrics_ramp_csv, subdir), index=False
        )
        pd.DataFrame(stats_records).to_csv(
            _paths(stats_csv, subdir), index=False
        )

        # ——— 3) PLOTTING ————————————————————————————————————————
        # comparison: constant vs all ramps
        fig1, (ax_o, ax_t) = plt.subplots(1, 2, figsize=(12,5))
        ax_o.plot(x_grid, U_const, "--", label="constant")
        for rr, U_r in zip(ramp_rates, profiles):
            ax_o.plot(x_grid, U_r, label=f"{rr} °C/min")
        ax_o.set_xlim(0, zoom_depth_nm)
        ax_o.set_xlabel("Depth (nm)"); ax_o.set_ylabel("O conc.")
        ax_o.set_title(f"O₂ @ {temp}°C, {hold}h")

        ax_t.plot(t_rh_h, T_K_rh - 273.15)
        ax_t.set_xlabel("Time (h)"); ax_t.set_ylabel("Temp (°C)")
        ax_t.set_title("T vs time")

        fig1.legend(loc="center right", title="Profile")
        fig1.tight_layout(rect=[0,0,0.85,1])
        fig1.savefig(_paths(plot_cmp_base.format(temp=temp, hold=hold), subdir), dpi=300)
        plt.close(fig1)

        # difference: (ramp – const)
        fig2, ax2 = plt.subplots(1,1,figsize=(6,5))
        for rr, U_r in zip(ramp_rates, profiles):
            ax2.plot(x_grid, U_r - U_const, label=f"{rr} °C/min")
        ax2.axhline(0, color="k", ls="--")
        ax2.set_xlim(0, zoom_depth_nm)
        ax2.set_xlabel("Depth (nm)")
        ax2.set_ylabel("Δ[O] (ramp−const)")
        ax2.set_title(f"Δ profile @ {temp}°C, {hold}h")
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(_paths(plot_diff_base.format(temp=temp, hold=hold), subdir), dpi=300)
        plt.close(fig2)

print("Done! CSVs and PDF plots written under:", os.path.abspath(exp_dir))
