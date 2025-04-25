import numpy as np
from scipy import constants, integrate, interpolate
import matplotlib.pyplot as plt
from model import c, u, v
from quantities import ell, lambda_eff, J, B
from cn_solver import CNSolver
from gle_solver import GLESolver
from dissolution_species import c_O
import os
import math


def gen_simulation_report(x, o_total, ell_val, lambda_eff_val, u_0, v_0, c_0, t, T, cc):

    # 5-panel figure to show the connected quantities
    fig, axes = plt.subplots(
    5,
    1,
    sharex=True,
    sharey=False,
    figsize=(4.8, 6.4 + 0.5 * 3.2),
    constrained_layout=True,
    )
    #fig.suptitle(f"Connected Quantities in Niobium Surface Baking\n"
    #             + f"Baking Conditions: $t = {t:.1f}$ h, $T = {T - 273.15:.1f}$ $^\\circ$C\n")




    # plot the oxygen diffusion profile
    axes[0].plot(
        x,
        o_total,
        "-",
        zorder=2,
        # color="C2",
        label=r"Oxygen Concentration",
    )
    # axes[0].plot(
    #     x,
    #     cc,
    #     "-",
    #     zorder=2,
    #     color="C2",
    #     label=r"Total: $c(x) = u(x) + v(x)$",
    # )
    axes[0].set_ylabel(r"$[\mathrm{O}]$ (at. %)")

    # annotate the plot with the simulation conditions
    # axes[0].text(
    #     0.6,
    #     0.9,
    #     f"$t = {t:.1f}$ h\n"
    #     + f"$T = {T - 273.15:.1f}$ $^\\circ$C\n",
    #     + f"$u_0 = {u_0:.1f}$ at. % nm\n"
    #     + f"$v_0 = {v_0:.1f}$ at. % nm\n"
    #     + f"$c_0 = {c_0:.2f}$ at. %",
    #     ha="left",
    #     va="top",
    #     fontsize="small",
    #     transform=axes[0].transAxes,
    # )
    # axes[0].text(
    #     0.55,
    #     0.9,
    #     "Nb baking conditions:",
    #     ha="right",
    #     va="top",
    #     fontsize="small",
    #     transform=axes[0].transAxes,
    # )
    axes[0].legend()



    # plot the depth-dependent mean-free-path
    line0, = axes[1].plot(
        x,
        ell_val,
        "-",
        zorder=1,
        label="Electron Mean-free-path",
    )
    hline_min = axes[1].axhline(ell_val.min(), linestyle=":", color="C1", zorder=0, label="Min $\ell$")
    hline_max = axes[1].axhline(ell_val.max(), linestyle=":", color="C2", zorder=0, label="Max $\ell$")
    axes[1].set_ylabel(r"$\ell$ (nm)")
    axes[1].set_ylim(0, None)
    axes[1].legend(handles=[line0, hline_min, hline_max])



    # Plot the depth-dependent penetration depth
    line1, = axes[2].plot(
        x,
        lambda_eff_val,
        "-",
        zorder=1,
        label="Magnetic Penetration depth",
    )

    # Add horizontal reference lines
    hline_min = axes[2].axhline(lambda_eff_val.min(), linestyle=":", color="C2", zorder=0, label="Min $\lambda_{\mathrm{eff.}}$")
    hline_max = axes[2].axhline(lambda_eff_val.max(), linestyle=":", color="C1", zorder=0, label="Max $\lambda_{\mathrm{eff.}}$")

    # Set axis labels
    axes[2].set_ylabel(r"$\lambda_{\mathrm{eff.}}$ (nm)")

    # Add legend including axhline handles
    #axes[2].legend(handles=[line1, hline_min, hline_max])
    axes[2].legend(handles=[line1, hline_max, hline_min])



    # initialize the GLE solver using the depth/penetration depth sampling points
    gle = GLESolver(
        x,
        lambda_eff_val,
    )

    # common arguments for the screening/currrent density profiles
    args = (100.0, 0.0, 0.0)
    screening_profile = gle.screening_profile(x, *args)
    B_max = B(x, args[0], lambda_eff_val.max())
    B_min = B(x, args[0], lambda_eff_val.min())

    # plot the Meissner screening profile
    axes[3].plot(
        x,
        screening_profile,
        "-",
        zorder=1,
        label="Screening profile",
    )
    axes[3].plot(
        x,
        B_max,
        ":",
        color="C1",
        zorder=0,
        #label="Max. $B(x)$",
    )
    axes[3].plot(
        x,
        B_min,
        ":",
        color="C2",
        zorder=0,
        #label="Min. $B(x)$",
    )
    axes[3].set_ylim(0, None)
    axes[3].set_ylabel("$B(x)$ (G)")
    axes[3].legend()

    current_density = gle.current_density(x, *args)
    J_max = J(x, args[0], lambda_eff_val.max())
    J_min = J(x, args[0], lambda_eff_val.min())
    # plot the current density
    axes[4].plot(
        x,
        current_density / 1e11,
        "-",
        zorder=1,
        label="Current density",
    )
    axes[4].plot(
        x,
        J_max / 1e11,
        ":",
        color="C1",
        zorder=0,
        #label="Max. $J(x)$",
    )
    axes[4].plot(
        x,
        J_min / 1e11,
        ":",
        color="C2",
        zorder=0,
        #label="Min. $J(x)$",
    )
    axes[4].set_ylim(0, None)
    axes[4].set_ylabel("$J(x)$ ($10^{11}$ A m$^{-2}$)")
    axes[4].legend()


    # axes[5].plot(
    #     x,
    #     ,
    #     "-",
    #     zorder=1,
    #     label="Critical current density",
    # )

    # label/limit the x-axis
    axes[-1].set_xlabel(r"$x$ (nm)")
    axes[-1].set_xlim(0, 150)



    # fig.savefig(f"sim_output/effect-on-sc-properties-t{t}-T{T}.pdf")
    # plt.close(fig)
 # Create a folder labeled by the simulation time and temperature
    folder_name = f"analysis/sim_t{t:.1f}_T{(T-273.15):.0f}"
    os.makedirs(folder_name, exist_ok=True)

    # Save the PDF in the simulation subfolder
    pdf_path = os.path.join(folder_name, f"effect-on-sc-properties-t{t:.1f}-T{(T-273.15):.0f}.pdf")
    fig.savefig(pdf_path)
    plt.close(fig)

    # -------------------------------
    # Now save the underlying data as CSV files:
    # -------------------------------

    # 1. Oxygen diffusion profile data: x, o_total, cc
    data = np.column_stack((x, o_total, cc))
    header = "x, o_total, cc"
    np.savetxt(os.path.join(folder_name, "oxygen_diffusion_profile.csv"),
               data, delimiter=",", header=header, comments='')

    # 2. Mean free path: x, ell_val
    data = np.column_stack((x, ell_val))
    header = "x, ell"
    np.savetxt(os.path.join(folder_name, "mean_free_path.csv"),
               data, delimiter=",", header=header, comments='')

    # 3. Penetration depth: x, lambda_eff_val
    data = np.column_stack((x, lambda_eff_val))
    header = "x, lambda_eff"
    np.savetxt(os.path.join(folder_name, "penetration_depth.csv"),
               data, delimiter=",", header=header, comments='')

    # 4. Screening profile: x, screening, B_max, B_min
    data = np.column_stack((x, screening_profile, B_max, B_min))
    header = "x, screening_profile, B_max, B_min"
    np.savetxt(os.path.join(folder_name, "screening_profile.csv"),
               data, delimiter=",", header=header, comments='')

    # 5. Current density: x, current_density, J_max, J_min
    data = np.column_stack((x, current_density, J_max, J_min))
    header = "x, current_density, J_max, J_min"
    np.savetxt(os.path.join(folder_name, "current_density.csv"),
               data, delimiter=",", header=header, comments='')
    #plt.show()

    return


def make_times(start=0.1, stop=96, step=0.5):
    """
    Generate times from `start` to `stop` (inclusive, if it fits exactly),
    with a chosen `step` size
    """
    times = [start]
    
    # Compute the first multiple of step that is greater than start.
    # If start is already a multiple of step, we don't want to repeat it.
    if math.isclose(start % step, 0, rel_tol=1e-9):
        next_time = start + step
    else:
        next_time = math.ceil(start / step) * step
    
    # Append subsequent multiples of step until stop is reached.
    while next_time <= stop:
        times.append(next_time)
        next_time += step
        
    return times


def make_temps_c2k(start_c=100, stop_c=400, step_c=10):
    """
    Generate a list of temperatures in Kelvin from `start_c` to `stop_c` (°C),
    using a chosen step size `step_c` (in °C).

    T(K) = T(°C) + 273.15
    """
    # Number of steps: floor division to ensure we don't go beyond `stop_c`
    num_steps = int((stop_c - start_c) / step_c)
    
    # Build the list in a single comprehension, ensuring we don't exceed `stop_c`
    temps_k = [
        273.15 + (start_c + i * step_c)
        for i in range(num_steps + 1)
        if (start_c + i * step_c) <= stop_c
    ]
    return temps_k


def main():

    # array of depth points in nm
    # note: the large endpoint is necessary for obtaining accurate solutions to the
    #       generalized london equation later on
    #x = np.linspace(0, 1000, 10001)
    times = make_times(0.1, 101, 1)
    temps = make_temps_c2k(100, 200, 20)

    times = [6]
    temps = [120+273.15]

    # simulation conditions for the low-temperature baking
    s_per_min = 60.0
    min_per_h = 60.0
    s_per_h = s_per_min * min_per_h
    #t = 12 * s_per_h  # s
    #T = 273.15 + 120  # K
    #t_h = 12 # time of simulation in hours
    u_0 = 1000.0  # at. % nm
    v_0 = 10.0  # at. % nm
    #c_0 = 0.01  # at. % nm


    DTYPE = np.double
    # specify the grid


    # simulate the components of the oxygen diffusion profile using Ciovati's
    # simplified model
    #o_surf = np.array([u(xx, t, T, u_0) for xx in x])
    #o_inter = np.array([v(xx, t, T, v_0, c_0) for xx in x])
    #c_total = np.array([c(xx, t, T, u_0, v_0, c_0) for xx in x_grid])
   
    for T in temps:
        for t in times:

            x_max = 1000.0  # boundary lengths: [0.0, x_max] (nm)
            N_x = 2001  # number of equally spaced grid points within spatial boundaries
            x_grid = np.linspace(0.0, x_max, N_x, dtype=DTYPE)

            t_max = t * s_per_h  # time domain lengths: [0.0, t_max] (s)
            N_t = 3001  # number of equally spaced grid points within temporal boundaries
            t_grid = np.linspace(0.0, t_max, N_t, dtype=DTYPE)
            
            cc = np.array([c(xx, t_max, T, u_0, v_0, 0.0) for xx in x_grid])
            c_0 = c_O(0, T, 100, 0, 0)  # at. % nm
            solver = CNSolver(T, u_0, v_0, t, x_max, N_x, N_t)
            U_record = solver.get_oxygen_profile()
            o_total = U_record[-1]

            # calculate the electron mean-free-path from the diffusion profile
            ell_val = ell(o_total)

            # calculate the effective penetration depth using the electron mean-free-path
            lambda_eff_val = lambda_eff(ell_val)

            gen_simulation_report(x_grid, o_total, ell_val, lambda_eff_val, u_0, v_0, c_0, t, T, cc)
            print(f"Simulation for T = {T-273.15} C and t = {t} h complete. simulation run is {solver.stability}")

    print("Simulations complete.")

    return

if __name__ == "__main__":
    main()