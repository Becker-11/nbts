from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from simulation.gle_solver import GLESolver
from simulation.quantities import ell, lambda_eff, lambda_eff_corr, J, B


class GenSimReport:
    """
    Modular report class for SRF simulation outputs.

    Usage:
        report = GenSimReport(x, o_total, t, T)
        report.compute()  # compute profiles
        fig = report.plot_overview()
    """

    def __init__(self, x, o_total, t, T, output_dir="sim_output"):
        # raw inputs
        self.x = np.asarray(x)
        self.o_total = np.asarray(o_total)
        self.t = t
        self.T = T
        self.lambda_0 = 27  # nm: clean-limit penetration depth
        # placeholders for computed arrays
        self.ell_val = None
        self.lambda_eff_val = None
        self.lambda_eff_val_corr = None
        self.screening_profile = None
        self.screening_profile_corr = None
        self.current_density = None
        self.current_density_corr = None
        self.B_clean = None
        self.B_dirty = None
        self.J_clean = None
        self.J_dirty = None
        # factors
        self.suppression_factor = None
        self.enhancement_factor = None
        self.current_suppression_factor = None
        # output directory and COMPUTE flag
        self.output_dir = Path(output_dir)
        self.COMPUTE = False

    def compute(self):
        """Compute all physics profiles from the oxygen concentration profile."""
        # mean-free-path and penetration depths
        self.ell_val = ell(self.o_total)
        self.lambda_eff_val = lambda_eff(self.ell_val)
        self.lambda_eff_val_corr = lambda_eff_corr(self.lambda_eff_val)

        # GLE solver for screening & current density
        gle = GLESolver(self.x, self.lambda_eff_val)
        gle_corr = GLESolver(self.x, self.lambda_eff_val_corr)
        # TODO: add applied field parameter to simulation
        args = (100.0, 0.0, 0.0)
        self.screening_profile = gle.screening_profile(self.x, *args)
        self.screening_profile_corr = gle_corr.screening_profile(self.x, *args)
        self.current_density = gle.current_density(self.x, *args)
        self.current_density_corr = gle_corr.current_density(self.x, *args)

        # reference B and J extrema via analytical formulas
        H0 = args[0]
        self.B_dirty = B(self.x, H0, self.lambda_eff_val.max())   # max B
        self.B_clean = B(self.x, H0, self.lambda_eff_val.min())   # min B
        self.J_dirty = J(self.x, H0, self.lambda_eff_val.max())   # max J
        self.J_clean = J(self.x, H0, self.lambda_eff_val.min())   # min J

        # derived factors
        J0 = self.J_clean[0]
        num = self.lambda_0 * J0
        den = np.clip(self.lambda_eff_val_corr * self.current_density_corr, 1e-10, None)
        self.suppression_factor = num / den
        self.enhancement_factor = self.lambda_eff_val_corr / self.lambda_0
        self.current_suppression_factor = self.current_density_corr / self.J_clean

        self.COMPUTE = True

    def _make_folder(self):
        folder = self.output_dir / f"sim_t{self.t:.1f}_T{int(self.T-273.15)}"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def save_report(self, fig, data_dict, tag):
        """
        Save a figure and its associated data arrays to disk.

        The figure is saved as {tag}.pdf in the 'plots' subfolder, and each array
        in data_dict is saved as {key}.csv in the 'data' subfolder of the simulation folder.
        """
        # base simulation folder
        folder = self._make_folder()
        # create subfolders for plots and data
        plots_folder = folder / "plots"
        data_folder = folder / "data"
        plots_folder.mkdir(parents=True, exist_ok=True)
        data_folder.mkdir(parents=True, exist_ok=True)

        # save figure to plots folder
        fig_path = plots_folder / f"{tag}.pdf"
        fig.savefig(fig_path)
        plt.close(fig)

        # save data arrays to data folder
        for key, arr in data_dict.items():
            arr = np.asarray(arr)
            if arr.ndim == 1:
                # stack x and y
                data_to_save = np.column_stack((self.x, arr))
                header = f"x,{key}"
            else:
                data_to_save = arr
                header = ",".join(f"col{i}" for i in range(arr.shape[1]))
            csv_path = data_folder / f"{key}.csv"
            np.savetxt(csv_path, data_to_save, delimiter=",", header=header, comments="")

    # -- Single-quantity plotters, overview quantities --
    def plot_oxygen(self, ax):
        ax.plot(self.x, self.o_total, '-', zorder=2, label='Oxygen Concentration')
        ax.set_ylabel(r'$[\mathrm{O}]$ (at. %)')
        ax.legend()

    def plot_mean_free_path(self, ax):
        line, = ax.plot(self.x, self.ell_val, '-', zorder=1, label='Electron Mean-free-path')
        hmin = ax.axhline(self.ell_val.min(), linestyle=':', label='Min $\ell$')
        hmax = ax.axhline(self.ell_val.max(), linestyle=':', label='Max $\ell$')
        ax.set_ylabel(r'$\ell$ (nm)')
        ax.set_ylim(0, None)
        ax.legend(handles=[line, hmin, hmax])

    def plot_penetration_depths(self, ax):
        l1, = ax.plot(self.x, self.lambda_eff_val, '-', label='Penetration depth')
        l2, = ax.plot(self.x, self.lambda_eff_val_corr, '-', label='Corrected penetration depth')
        ax.axhline(self.lambda_eff_val.min(), linestyle=':', label='Min $\lambda_{eff}$')
        ax.axhline(self.lambda_eff_val.max(), linestyle=':', label='Max $\lambda_{eff}$')
        ax.axhline(self.lambda_eff_val_corr.max(), linestyle=':', label='Max $\lambda_{eff}$ (corr)')
        ax.set_ylabel(r'$\lambda_{eff}$ (nm)')
        ax.legend(handles=[l1, l2])

    def plot_screening(self, ax):
        ax.plot(self.x, self.screening_profile, '-', zorder=1, label='Screening profile')
        ax.plot(self.x, self.screening_profile_corr, '-', zorder=1, label='Corrected screening profile')
        ax.plot(self.x, self.B_dirty, ':', zorder=0, label='B(x) dirty')
        ax.plot(self.x, self.B_clean, ':', zorder=0, label='B(x) clean')
        ax.set_ylabel(r'$B(x)$ (G)')
        ax.set_ylim(0, None)
        ax.legend()

    def plot_current(self, ax):
        ax.plot(self.x, self.current_density/1e11, '-', zorder=1, label='Current density')
        ax.plot(self.x, self.current_density_corr/1e11, '-', zorder=1, label='Corrected current density')
        ax.plot(self.x, self.J_dirty/1e11, ':', zorder=0, label='J(x) dirty')
        ax.plot(self.x, self.J_clean/1e11, ':', zorder=0, label='J(x) clean')
        ax.set_ylabel(r'$J(x)$ ($10^{11}$ A m$^{-2}$)')
        ax.set_ylim(0, None)
        ax.legend()

    # -- Single-quantity suppression/enhancement plotters --
    def plot_suppression(self, ax, invert=False):
        if invert:
            ax.plot(self.x, 1 / self.suppression_factor, '-', label='Suppression factor (inverse)')
        else:
            ax.plot(self.x, self.suppression_factor, '-', label='Suppression factor')
        ax.set_ylabel('Suppression factor')
        ax.legend()

    def plot_enhancement(self, ax):
        ax.plot(self.x, self.enhancement_factor, '-', label='Penetration depth enhancement')
        ax.set_ylabel(r'$\lambda(x)/\lambda_{clean}$')
        ax.legend()

    def plot_current_suppression(self, ax):
        ax.plot(self.x, self.current_suppression_factor, '-', label='Current density suppression')
        ax.set_ylabel(r'$J(x)/J_{clean}(x)$')
        ax.set_xlabel(r'$x$ (nm)')
        ax.legend()

    def plot_ratio(self, ax):
        ratio = self.enhancement_factor * self.current_suppression_factor
        ax.plot(self.x, ratio, '-', label='Ratio of enhancement and suppression')
        ax.set_ylabel(r'$\lambda(x)/\lambda_{clean} \cdot J(x)/J_{clean}(x)$')
        ax.set_xlabel(r'$x$ (nm)')
        ax.legend()

    # -- Assemble plots --
    def plot_overview(self):
        if not self.COMPUTE:
            self.compute()
        fig, axes = plt.subplots(5, 1,
                                 sharex=True, sharey=False,
                                 figsize=(4.8, 6.4 + 0.5 * 3.2),
                                 constrained_layout=True)
        # panels
        self.plot_oxygen(axes[0])
        self.plot_mean_free_path(axes[1])
        self.plot_penetration_depths(axes[2])
        self.plot_screening(axes[3])
        self.plot_current(axes[4])
        # final formatting
        axes[-1].set_xlabel(r'$x$ (nm)')
        axes[-1].set_xlim(0, 150)
        plt.suptitle(f"Simulation overview for T = {self.T-273.15:.1f} C and t = {self.t:.1f} h")
        # save data
        data = {
            'oxygen_diffusion_profile':    self.o_total,
            'mean_free_path':              self.ell_val,
            'penetration_depth':           self.lambda_eff_val,
            'penetration_depth_corrected': self.lambda_eff_val_corr,
            'screening_profile':           self.screening_profile,
            'screening_profile_corrected': self.screening_profile_corr,
            'current_density':             self.current_density,
            'current_density_corrected':   self.current_density_corr,
        }
        self.save_report(fig, data, tag='overview')
        return fig

    def plot_suppression_factor_comparison(self):
        if not self.COMPUTE:
            self.compute()
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(5, 8), constrained_layout=True)
        self.plot_suppression(axes[0], True)
        self.plot_enhancement(axes[1])
        self.plot_current_suppression(axes[2])
        self.plot_ratio(axes[3])
        axes[-1].set_xlim(0, 150)
        plt.suptitle(f"Simulation suppression factor for T = {self.T-273.15:.1f} C and t = {self.t:.1f} h")
        data = {
            'suppression_factor':          self.suppression_factor,
            'enhancement_factor':          self.enhancement_factor,
            'current_suppression_factor':  self.current_suppression_factor
        }
        self.save_report(fig, data, tag='suppression_factor_comparison')
        return fig

    def plot_suppression_factor(self):
        """Single-panel suppression-factor vs x."""
        if not self.COMPUTE:
            self.compute()
        fig, ax = plt.subplots()
        self.plot_suppression(ax)
        ax.set_ylim(0, 5)
        ax.set_xlim(0, 40)
        ax.set_xlabel(r'$x$ (nm)')
        ax.set_title(f"Suppression factor at T={self.T-273.15:.1f}\u00B0C, t={self.t:.1f}h")
        data = {'suppression_factor': self.suppression_factor}
        self.save_report(fig, data, tag='suppression_factor')
        return fig
    
