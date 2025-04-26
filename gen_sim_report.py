'''
Class to generate plots from a single simulation run.
'''

import matplotlib.pyplot as plt
import numpy as np
import os
from gle_solver import GLESolver
from quantities import ell, lambda_eff, lambda_eff_corr, J, B

class GenSimReport:
    def __init__(self, x_grid, o_total, t, T):
        self.x_grid = x_grid
        self.o_total = o_total
        self.t = t
        self.T = T

        # these will be filled in by compute_profiles()
        self.ell_val            = None
        self.lambda_eff_val     = None
        self.lambda_eff_val_corr= None
        self.screening_profile  = None
        self.screening_profile_corr = None
        self.current_density    = None
        self.current_density_corr = None
        self.B_max              = None
        self.B_min              = None
        self.J_max              = None
        self.J_min              = None



    def compute_profiles(self):
        # 1) compute all the “physics” arrays once
        self.ell_val             = ell(self.o_total)
        self.lambda_eff_val      = lambda_eff(self.ell_val)
        self.lambda_eff_val_corr = lambda_eff_corr(self.lambda_eff_val)

        # set up solvers
        gle      = GLESolver(self.x_grid, self.lambda_eff_val)
        gle_corr = GLESolver(self.x_grid, self.lambda_eff_val_corr)

        args = (100.0, 0.0, 0.0)
        self.screening_profile      = gle.screening_profile(self.x_grid, *args)
        self.screening_profile_corr = gle_corr.screening_profile(self.x_grid, *args)
        self.current_density        = gle.current_density(self.x_grid, *args)
        self.current_density_corr   = gle_corr.current_density(self.x_grid, *args)

        # extrema for reference lines
        self.B_max = B(self.x_grid, args[0], self.lambda_eff_val.max())
        self.B_min = B(self.x_grid, args[0], self.lambda_eff_val.min())
        # TODO: re-name J_max, J_min and consider lambda_eff_vall_corr lines
        self.J_max = J(self.x_grid, args[0], self.lambda_eff_val.max())
        self.J_min = J(self.x_grid, args[0], self.lambda_eff_val.min())


    def plot(self):
        # Ensure all profiles are computed
        if self.ell_val is None:
            self.compute_profiles()

        # Create a 5-panel figure
        fig, axes = plt.subplots(
            5, 1,
            sharex=True, sharey=False,
            figsize=(4.8, 6.4 + 0.5 * 3.2),
            constrained_layout=True
        )

        # 1) Oxygen diffusion profile
        axes[0].plot(
            self.x_grid,
            self.o_total,
            '-', zorder=2,
            label=r'Oxygen Concentration'
        )
        axes[0].set_ylabel(r'$[\mathrm{O}]$ (at. %)')
        axes[0].legend()

        # 2) Electron mean-free-path
        line0, = axes[1].plot(
            self.x_grid,
            self.ell_val,
            '-', zorder=1,
            label=r'Electron Mean-free-path'
        )
        hline_min = axes[1].axhline(
            self.ell_val.min(), linestyle=':', color='C1', zorder=0,
            label=r'Min $\ell$'
        )
        hline_max = axes[1].axhline(
            self.ell_val.max(), linestyle=':', color='C2', zorder=0,
            label=r'Max $\ell$'
        )
        axes[1].set_ylabel(r'$\ell$ (nm)')
        axes[1].set_ylim(0, None)
        axes[1].legend(handles=[line0, hline_min, hline_max])

        # 3) Magnetic penetration depths
        line1, = axes[2].plot(
            self.x_grid,
            self.lambda_eff_val,
            '-', zorder=1,
            label='Magnetic Penetration depth'
        )
        line2, = axes[2].plot(
            self.x_grid,
            self.lambda_eff_val_corr,
            '-', zorder=1,
            label=r'Corrected $\lambda_{\mathrm{eff.}}$'
        )
        hline_min = axes[2].axhline(
            self.lambda_eff_val.min(), linestyle=':', color='C2', zorder=0,
            label=r'Min $\lambda_{\mathrm{eff.}}$'
        )
        hline_max = axes[2].axhline(
            self.lambda_eff_val.max(), linestyle=':', color='C1', zorder=0,
            label=r'Max $\lambda_{\mathrm{eff.}}$'
        )
        hline_max_corr = axes[2].axhline(
            self.lambda_eff_val_corr.max(), linestyle=':', color='C3', zorder=0,
            label=r'Max $\lambda_{\mathrm{eff.}}$ (corr)'
        )
        axes[2].set_ylabel(r'$\lambda_{\mathrm{eff.}}$ (nm)')
        axes[2].legend(handles=[line1, line2, hline_max, hline_min, hline_max_corr])

        # 4) Meissner screening profiles
        axes[3].plot(
            self.x_grid,
            self.screening_profile,
            '-', zorder=1,
            label='Screening profile'
        )
        axes[3].plot(
            self.x_grid,
            self.screening_profile_corr,
            '-', zorder=1,
            label='Corrected screening profile'
        )
        axes[3].plot(self.x_grid, self.B_max, ':', color='C1', zorder=0)
        axes[3].plot(self.x_grid, self.B_min, ':', color='C2', zorder=0)
        axes[3].set_ylim(0, None)
        axes[3].set_ylabel(r'$B(x)$ (G)')
        axes[3].legend()

        # 5) Current density profiles
        axes[4].plot(
            self.x_grid,
            self.current_density / 1e11,
            '-', zorder=1,
            label='Current density'
        )
        axes[4].plot(
            self.x_grid,
            self.current_density_corr / 1e11,
            '-', zorder=1,
            label='Corrected current density'
        )
        axes[4].plot(self.x_grid, self.J_max / 1e11, ':', color='C1', zorder=0)
        axes[4].plot(self.x_grid, self.J_min / 1e11, ':', color='C2', zorder=0)
        axes[4].set_ylim(0, None)
        axes[4].set_ylabel(r'$J(x)$ ($10^{11}$ A m$^{-2}$)')
        axes[4].legend()

        # Final x-axis label and limit
        axes[-1].set_xlabel(r'$x$ (nm)')
        axes[-1].set_xlim(0, 150)

        return fig




    def save_data(self, fig=None):
        if self.ell_val is None:
            self.compute_profiles()
        if fig is None:
            fig = self.plot()
        # TODO: see if folder_root can be a parameter of save_data
        # folder_name = os.path.join(
        #     folder_root,
        #     f"/sim_t{self.t:.1f}_T{(self.T-273.15):.0f}"
        # )
        folder_name = f"analysis/sim_t{self.t:.1f}_T{(self.T-273.15):.0f}"
        os.makedirs(folder_name, exist_ok=True)

        # save figure
        pdf_path = os.path.join(
            folder_name,
            f"effect-on-sc-properties-t{self.t:.1f}-T{(self.T-273.15):.0f}.pdf"
        )
        fig.savefig(pdf_path)
        plt.close(fig)

        # now save each csv, *using* the exact attributes you already have:
        profiles = {
            "oxygen_diffusion_profile.csv": np.column_stack((self.x_grid, self.o_total)),
            "mean_free_path.csv":          np.column_stack((self.x_grid, self.ell_val)),
            "penetration_depth.csv":       np.column_stack((self.x_grid, self.lambda_eff_val)),
            "screening_profile.csv":       np.column_stack((self.x_grid, self.screening_profile,
                                                            self.B_max, self.B_min)),
            "current_density.csv":         np.column_stack((self.x_grid, self.current_density,
                                                            self.J_max, self.J_min)),
        }
        for fname, data in profiles.items():
            header = ",".join(["x"] + fname.replace(".csv","").split("_"))
            np.savetxt(
                os.path.join(folder_name, fname),
                data,
                delimiter=",",
                header=header,
                comments="",
            )
