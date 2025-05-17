# nbts/analysis/simulation_analyzer.py
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ────────────────────────────────────────────────────────────────────────
# Shared scaffolding
# ────────────────────────────────────────────────────────────────────────
@dataclass
class Metric:
    """Everything needed to render one heat‑map plot."""
    z_grid:      np.ndarray
    cmap:        str
    levels:      int
    cbar_label:  str
    fname:       str


class SimulationAnalyzer(ABC):
    """
    Base class: concrete subclasses implement `_collect()`, then call `run()`.
    """

    def __init__(self, base_dir: str, ox_dir: str | None = None) -> None:
        self.base_dir     = Path(base_dir)
        self.ox_dir       = Path(ox_dir) if ox_dir else None
        self.analysis_dir = self.base_dir.parent / f"analysis_{self.ox_dir.name}" if self.ox_dir else self.base_dir.parent / "analysis"
        self.analysis_dir.mkdir(exist_ok=True)

        self.times:   np.ndarray | None = None
        self.temps:   np.ndarray | None = None
        self.metrics: dict[str, Metric] = {}

    # public entry point ---------------------------------------------------
    def run(self) -> None:
        self._collect()
        self._make_plots()
        print(f"{self.__class__.__name__}: analysis complete → {self.analysis_dir}")

    # to be supplied by subclass ------------------------------------------
    @abstractmethod
    def _collect(self) -> None: ...

    # common plot loop -----------------------------------------------------
    def _make_plots(self) -> None:
        X, Y = np.meshgrid(self.times, self.temps)

        for m in self.metrics.values():
            fig, ax = plt.subplots(figsize=(8, 6))
            mesh = ax.pcolormesh(X, Y, m.z_grid, shading="gouraud", cmap=m.cmap)

            cbar = fig.colorbar(mesh, ax=ax)
            cbar.set_label(m.cbar_label, fontsize=12)

            cs = ax.contour(X, Y, m.z_grid,
                            levels=m.levels, colors="white", linewidths=1)
            ax.clabel(cs, inline=True, fontsize=8)

            ax.set_xlabel("Time (h)", fontsize=14)
            ax.set_ylabel("Temperature (°C)", fontsize=14)
            fig.tight_layout()
            fig.savefig(self.analysis_dir / m.fname)
            plt.close(fig)


# ────────────────────────────────────────────────────────────────────────
# Current‑density overview (concrete subclass)
# ────────────────────────────────────────────────────────────────────────
class CurrentDensityAnalyzer(SimulationAnalyzer):

    # helpers --------------------------------------------------------------
    @staticmethod
    def _parse_time_temp(name: str) -> tuple[float | None, float | None]:
        try:
            _, t_tag, T_tag = name.split("_")
            return float(t_tag[1:]), float(T_tag[1:])
        except (ValueError, IndexError):
            return None, None

    @staticmethod
    def _surface(df: pd.DataFrame, col: str) -> float:
        s = df.loc[df["x"] == 0, col]
        if s.empty:
            s = df.loc[[df["x"].idxmin()], col]
        return s.iloc[0]

    # core ---------------------------------------------------------------
    def _collect(self) -> None:
        folders = glob.glob(str(self.base_dir / "sim_t*_T*"))

        t_all, T_all, A_all, B_all, C_all = ([] for _ in range(5))

        for f in folders:
            t_val, T_val = self._parse_time_temp(Path(f).name)
            if t_val is None:
                continue

            csv = Path(f) / "data" / "current_density_corrected.csv"
            if not csv.is_file():
                continue

            df = pd.read_csv(csv, skipinitialspace=True)
            if not {"x", "current_density_corrected", "J_clean_corr"}.issubset(df.columns):
                continue

            # Metric A
            J_max     = df["current_density_corrected"].max()
            J_surface = self._surface(df, "J_clean_corr")
            ratio_A   = J_max / J_surface

            # Metric B
            x_peak    = df.loc[df["current_density_corrected"].idxmax(), "x"]

            # Metric C
            J_max_surf = self._surface(df, "current_density_corrected")
            ratio_C    = J_max_surf / J_surface

            t_all.append(t_val);   T_all.append(T_val)
            A_all.append(ratio_A); B_all.append(x_peak); C_all.append(ratio_C)

        # turn lists into 2‑D grids --------------------------------------
        self.times = np.unique(t_all)
        self.temps = np.unique(T_all)
        shape      = (len(self.temps), len(self.times))
        Z_A, Z_B, Z_C = (np.full(shape, np.nan) for _ in range(3))

        for t, T, a, b, c in zip(t_all, T_all, A_all, B_all, C_all):
            i = np.where(self.times == t)[0][0]
            j = np.where(self.temps == T)[0][0]
            Z_A[j, i], Z_B[j, i], Z_C[j, i] = a, b, c

        # register the three plots 
        self.metrics = {
            "ratio_A": Metric(
                Z_A, "cividis", 10,
                "maximum J(x) / maximum supercurrent in clean Nb",
                "ratio_max_over_surface.pdf",
            ),
            "x_peak": Metric(
                Z_B, "cividis", 3,
                "x position of supercurrent density peak",
                "x_peak_position.pdf",
            ),
            "ratio_C": Metric(
                Z_C, "cividis", 10,
                "J(x) surface value / maximum supercurrent in clean Nb",
                "surface_current_ratio.pdf",
            ),
        }
