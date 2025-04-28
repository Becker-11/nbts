import numpy as np
from cn_solver import CNSolver
from gen_sim_report import GenSimReport
from dissolution_species import c_O
import math



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

    for T in temps:
        for t in times:

            x_max = 1000.0  # boundary lengths: [0.0, x_max] (nm)
            N_x = 2001  # number of equally spaced grid points within spatial boundaries
            x_grid = np.linspace(0.0, x_max, N_x, dtype=DTYPE)

            t_max = t * s_per_h  # time domain lengths: [0.0, t_max] (s)
            N_t = 3001  # number of equally spaced grid points within temporal boundaries
            t_grid = np.linspace(0.0, t_max, N_t, dtype=DTYPE)

            # TODO: add initial concentration from dissolution_species 
            solver = CNSolver(T, u_0, v_0, t, x_max, N_x, N_t)
            U_record = solver.get_oxygen_profile()
            o_total = U_record[-1]

            report = GenSimReport(x_grid, o_total, t, T)
            report.plot_overview()
            #report.plot()
            #report.plot_potential_increase()
            #report.save_data()
            print(f"Simulation for T = {T-273.15} C and t = {t} h complete. simulation run is {solver.stability}")

    print("Simulations complete.")

    return

if __name__ == "__main__":
    main()