#!/usr/bin/env python3
"""
Custom plotting for paper figures
──────────────────────────────────────────────────
* Re‑creates the three heat‑maps of CurrentDensityAnalyzer but lets you
  change any styling without affecting the main nbts pipeline.
* Adds plot_single_sim() → for single recipe plots.

"""

# ─── CONFIG SECTION ─────────────────────────────────────────────────────
BASE_DIR     = "experiments/2025-05-24_const_e10b3cd"  # path to the main experiment folder
RESULTS_DIR   = "results"     # subfolder inside BASE_DIR
SIM_DIR      = "sim_t8.0_T125"  # subfolder inside RESULTS_DIR, for single sim plots
OUTPUT_DIR   = "figures"     # created inside BASE_DIR/..
COLORMAP     = "cividis"             # change freely
LEVELS_A     = 5                    # contour levels for each heat‑map
LEVELS_B     =  1
LEVELS_C     = 6
DPI          = 400                   # PDF/PNG output resolution
# ────────────────────────────────────────────────────────────────────────

from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, CubicSpline
from scipy.ndimage import gaussian_filter

# ─── helpers ------------------------------------------------------------
def parse_time_temp(dirname: str):
    """Extract numerical t and T from a folder name like 'sim_t6.0_T115'."""
    try:
        _, t_tag, T_tag = dirname.split("_")
        return float(t_tag[1:]), float(T_tag[1:])
    except ValueError:
        return None, None


def surface(df: pd.DataFrame, col: str):
    """Return value at x=0 (or the nearest available point)."""
    s = df.loc[df["x"] == 0, col]
    if s.empty:
        s = df.loc[[df["x"].idxmin()], col]
    return s.iloc[0]

# ─── main heat‑map routine ------------------------------------------------
def plot_overview_heatmaps(base_dir: Path, out_dir: Path, time, temp):
    folders = glob.glob(str(base_dir / "results" / "sim_t*_T*"))

    t_vals, T_vals, Z_A, Z_B, Z_C = [], [], [], [], []

    for f in folders:
        t, T = parse_time_temp(Path(f).name)
        if t is None:
            continue

        df_path = Path(f) / "data" / "smooth_current_density.csv"
        if not df_path.is_file():
            continue
        df = pd.read_csv(df_path, skipinitialspace=True)
        if not {"x", "current_density_smooth", "J_clean"}.issubset(df.columns):
            continue

        # --- sub‑grid peak on a cubic spline -----------------------------
        x  = df["x" ].to_numpy()
        J  = df["current_density_smooth"].to_numpy()
        spl = CubicSpline(x, J, bc_type="natural")
        fine_x = np.linspace(x.min(), x.max(), 40001)
        J_fine = spl(fine_x)
        idx_max = J_fine.argmax()
        x_peak = float(fine_x[idx_max])          # smooth peak position
        J_max  = float(J_fine[idx_max])          # smooth peak value

        J_surf_clean  = surface(df, "J_clean")   # reference @ x = 0
        J_surf_smooth = float(spl(0.0))          # smooth J at surface

        ratioA = J_max         / J_surf_clean
        ratioC = J_surf_smooth / J_surf_clean

        t_vals.append(t);   T_vals.append(T)
        Z_A.append(ratioA); Z_B.append(x_peak); Z_C.append(ratioC)

    # --------------------------- remainder unchanged ----------------------
    times = np.unique(t_vals)
    temps = np.unique(T_vals)
    shape = (len(temps), len(times))
    A = np.full(shape, np.nan)
    B = np.full(shape, np.nan)
    C = np.full(shape, np.nan)

    for t, T, a, b, c in zip(t_vals, T_vals, Z_A, Z_B, Z_C):
        i = np.where(times == t)[0][0]
        j = np.where(temps == T)[0][0]
        A[j, i], B[j, i], C[j, i] = a, b, c

    heatmap_specs = [
        ("ratio_max_over_surface.pdf", A,
         r'$\tilde{J}\,\equiv\; \frac{\max\{J(x)\}}{\max\{J_{\mathrm{clean}}(x)\}}$', LEVELS_A),
        ("x_peak_position.pdf", B,
         r'$\tilde{x}\;\equiv\;\operatorname{arg\,max}_{x}\,\{J(x)\}$', LEVELS_B),
        ("surface_current_ratio.pdf", C,
         r'$\tilde{J}_0 \;\equiv\; \dfrac{J(x=0)}{\max\{J_{\mathrm{clean}}(x)\}}$', LEVELS_C),
    ]

    ti = np.linspace(times.min(), times.max(), len(times)*10)
    Ti = np.linspace(temps.min(), temps.max(), len(temps)*10)
    Xd, Yd = np.meshgrid(ti, Ti)

    def valid_points(Z2d):
        jj, ii = np.nonzero(~np.isnan(Z2d))
        return np.column_stack([times[ii], temps[jj]]), Z2d[jj, ii]

    for fname, Z2d, cbar_label, n_levels in heatmap_specs:
        # ── choose grid (no interpolation shown here) ──────────────────────
        Z_plot             = Z2d
        X_plot, Y_plot     = np.meshgrid(times, temps)

        # ── build figure WITHOUT constrained_layout ───────────────────────
        fig, ax = plt.subplots(figsize=(6.4, 4.8))   # ← no constrained_layout

        # heat‑map + contour
        mesh = ax.pcolormesh(X_plot, Y_plot, Z_plot,
                            cmap=COLORMAP, shading='gouraud')
        cs   = ax.contour(X_plot, Y_plot, Z_plot,
                        levels=n_levels, colors='w', linewidths=0.8)
        ax.clabel(cs, inline=True, fontsize=7)

        # colour‑bar
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label(cbar_label)

        # ── coordinates for annotation outside the colour‑bar ─────────────
        cbox   = cbar.ax.get_position()         # Bbox in figure‑fraction coords
        bbox_ax = ax.get_position()     # (x0, y0, x1, y1) in figure coords

        x_lab = 0.25                    # centred horizontally in the figure
        y_lab = 0.5 * (bbox_ax.y1 + 1)  # halfway between axes‑top and figure‑top

        # recipe marker
        ax.scatter(time, temp, s=20, facecolor='firebrick', lw=1.3, zorder=4)

        # centred label + straight arrow
        ax.annotate(
            rf'{time:.1f} h, {temp:.0f} °C',
            xy=(time + 0.5, temp + 1.5), xycoords='data',            # arrow tail
            xytext=(x_lab, y_lab), textcoords='figure fraction',
            ha='center', va='center',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3',
                    fc='white', alpha=0.85, lw=0.8),
            arrowprops=dict(arrowstyle='->', lw=0.8, color='firebrick',
                            shrinkA=0, shrinkB=0),
            zorder=5
        )

        # labels and save
        ax.set_xlabel(r'$t$ (h)')
        ax.set_ylabel(r'$T$ ($^{\circ}$C)')
        fig.tight_layout()                 # tidy margins (ignores annotation)
        fig.savefig(out_dir / fname, dpi=DPI)
        plt.close(fig)




# ────────────────────────────────────────────────────────────────────────
# Data loader ------------------------------------------------------------
def get_data(sim_dir: Path) -> dict[str, np.ndarray]:
    """
    Read every CSV in <sim_dir>/data/ into a dict  {name: ndarray}.
    Also compute B_dirty/clean, J_dirty/clean, etc. that the plotters need.
    """
    dpath = sim_dir / "data"
    if not dpath.is_dir():
        raise FileNotFoundError(f"{dpath} not found")

    data = {}
    for csv in dpath.glob("*.csv"):
        key = csv.stem                         # e.g. 'current_density'
        df  = pd.read_csv(csv, skipinitialspace=True)
        data[key] = df

    # shorthand arrays ---------------------------------------------------
    data['x_short'] = data["current_density"]["x"].to_numpy()
    data["x"] = data["smooth_current_density"]["x"].to_numpy()

    # Oxygen total (at. %) ----------------------------------------------
    if "oxygen_diffusion_profile" in data:
        data["o_total"] = data["oxygen_diffusion_profile"]["oxygen_diffusion_profile"].to_numpy()

    # Mean‑free path -----------------------------------------------------
    if "mean_free_path" in data:
        data["ell_val"] = data["mean_free_path"]["mean_free_path"].to_numpy()

    # Penetration depths -------------------------------------------------
    if "penetration_depth_corrected" in data:
        data["lambda_eff_val"] = data["penetration_depth_corrected"]["penetration_depth_corrected"].to_numpy()

    # Screening & B fields ----------------------------------------------
    if "screening_profile_corrected" in data:
        data["screening_profile"] = data["screening_profile_corrected"]["screening_profile_corrected"].to_numpy()
        data["B_clean"] = data["screening_profile_corrected"]["B_clean_corr"].to_numpy()
        data["B_dirty"] = data["screening_profile_corrected"]["B_dirty_corr"].to_numpy()

    if "smooth_current_density" in data:
        data["current_density"] = data["smooth_current_density"]["current_density_smooth"].to_numpy()
        data["J_clean"] = data["smooth_current_density"]["J_clean"].to_numpy()
        data["J_dirty"] = data["smooth_current_density"]["J_dirty"].to_numpy()
    
    if "smooth_critical_current_density" in data:
        data["J_c"] = data["smooth_critical_current_density"]["critical_current_density_smooth"].to_numpy()

    # Metadata -----------------------------------------------------------
    t, T = parse_time_temp(sim_dir.name)
    data["t"] = t
    data["T"] = T
    return data


# ────────────────────────────────────────────────────────────────────────
# Stateless small plotters ----------------------------------------------
def _plot_currents(ax, d):
    ax.plot(d["x"], d["current_density"]/1e11, label=r'$J(x)$')
    ax.plot(d["x"], d["J_dirty"]/1e11,  ':', label=r'$J_\mathrm{dirty}(x)$')
    ax.plot(d["x"], d["J_clean"]/1e11,  ':', label=r'$J_\mathrm{clean}(x)$')
    ax.plot(d["x"], d["J_c"]/1e11, label=r'$J_c(x)$')
    ax.set_ylim(0, None)
    ax.legend(frameon=False)

def _plot_critical(ax, d):
    if "J_c" in d:
        ax.plot(d["x"], d["J_c"]/1e11, label=r'$J_c(x)$')
        ax.set_ylim(0, None)
        ax.legend()

def _plot_current_ratio(ax, d):
    ratio = d["current_density"] / d["J_c"]
    ax.plot(d["x"], ratio, label=r'$J(x)$ / $J_c$')
    ax.set_ylabel(r'$j(x) \equiv \frac{ J(x) }{ J_{c}(x) }$')
    ax.set_ylim(0, None)
    #ax.legend()

# High‑level wrapper -----------------------------------------------------
def plot_current_densities(sim_dir: Path, out_dir: Path, d):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(4.8, 4.8),
                             constrained_layout=True)
    _plot_currents(axes[0], d)
    _plot_current_ratio(axes[1], d)
    axes[0].set_ylabel(r'Current Densities ($10^{11}$ A m$^{-2}$)')
    axes[-1].set_xlim(0, 150)
    axes[0].set_ylim(0, 6)
    axes[-1].set_xlabel(r"$x$ (nm)")
    #plt.suptitle(f"T = {d['T']:.1f} °C,  t = {d['t']:.1f} h")
    fname = sim_dir.name + "_current_density.pdf"
    fig.savefig(out_dir / fname, dpi=DPI)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────
# Overview quantities (oxygen, mean‑free‑path, λ, screening) ------------
def _plot_oxygen(ax, d):
    if "o_total" in d:
        ax.plot(d["x_short"], d["o_total"], label=r'Oxygen Concentration')
        ax.set_xlim(0, 150)
        ax.set_ylabel(r'[O] at.%')
        #ax.legend()

def _plot_mfp(ax, d):
    if "ell_val" in d:
        line, = ax.plot(d["x_short"], d["ell_val"], label=r'Electron Mean‑free‑path')
        ax.plot(d['x_short'], np.full(len(d["x_short"]), d["ell_val"].min()), linestyle=':', zorder=1)
        ax.plot(d['x_short'], np.full(len(d["x_short"]), d["ell_val"].max()), linestyle=':', zorder=1)
        ax.set_ylabel(r'$\ell$ (nm)')
        ax.set_xlim(0, 150)
        ax.set_ylim(0, None)
        # ax.legend(
        #     loc='upper right',          # anchor = upper‑right corner of the box
        #     bbox_to_anchor=(1, 0.92),   # (x, y) in axes coords → move down to 92 %
        #     frameon=True, framealpha=0.9
        # )

def _plot_pen_depth(ax, d):
        line, = ax.plot(d["x_short"], d["lambda_eff_val"], label=r'Magnetic Penetration Depth')
        ax.plot(d["x_short"], np.full(len(d["x_short"]), d["lambda_eff_val"].min()), linestyle=':', zorder=1)
        ax.plot(d["x_short"], np.full(len(d["x_short"]), d["lambda_eff_val"].max()), linestyle=':', zorder=1)
        ax.set_xlim(0, 150)
        ax.set_ylabel(r'$\lambda$ (nm)')
        # ax.legend(
        #     loc='upper right',          # anchor = upper‑right corner of the box
        #     bbox_to_anchor=(1, 0.92),   # (x, y) in axes coords → move down to 92 %
        #     frameon=True, framealpha=0.9
        # )

def _plot_screening(ax, d):
    if "screening_profile" in d:
        ax.plot(d["x_short"], d["screening_profile"], label='Magnetic Screening Profile')
        ax.plot(d["x_short"], d["B_dirty"], linestyle=':', label=r'$B$ dirty', zorder=1)
        ax.plot(d["x_short"], d["B_clean"], linestyle=':', label=r'$B$ clean', zorder=1)
        ax.set_ylabel(r'$B(x)$ (G)')
        ax.set_xlim(0, 150)
        ax.set_ylim(0, None)
        #ax.legend()

def plot_overview_quantities(sim_dir: Path, out_dir: Path, d):
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(4.8, 6.8),
                             constrained_layout=True)

    _plot_oxygen   (axes[0], d)
    _plot_mfp      (axes[1], d)
    _plot_pen_depth(axes[2], d)
    _plot_screening(axes[3], d)


    axes[-1].set_xlabel(r"$x$ (nm)")
    #fig.suptitle(f"Overview,  T = {d['T']:.1f} °C,  t = {d['t']:.1f} h")

    fname = sim_dir.name + "_overview.pdf"
    fig.savefig(out_dir / fname, dpi=DPI)
    plt.close(fig)



# ─── running block ------------------------------------------------------
def main():
    base = Path(BASE_DIR)
    out  = base / OUTPUT_DIR
    out.mkdir(exist_ok=True)
    sim_dir = base / RESULTS_DIR / SIM_DIR
    data = get_data(sim_dir)

    # overview heat‑maps
    plot_overview_heatmaps(base, out, data["t"], data["T"])
    print(f"Heat‑maps written → {out}")


    # single simulation profile

    plot_current_densities(sim_dir, out, data)
    plot_overview_quantities(sim_dir, out, data)
    print(f"Example profile written → {out}")
    print(f"single simulation overview written → {out}")

if __name__ == "__main__":
    main()
