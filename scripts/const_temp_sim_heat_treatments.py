import numpy as np
import yaml
import argparse
import math
import os

from solvers.cn_solver_const_temp import CNSolver
from utils.gen_sim_report import GenSimReport

from test_sim_heat_treatments import test_oxygen_profile


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


def load_sim_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # build sim‐run durations (in hours) and T list (in K)
    times = make_times(**cfg["time"])
    temps = make_temps_c2k(**cfg["temperature"])
    # initial surface amounts
    u0, v0 = cfg["initial"]["u0"], cfg["initial"]["v0"]
    # grid specs
    grid = cfg["grid"]
    x_max, n_x, n_t = grid["x_max_nm"], grid["n_x"], grid["t_steps"]
    return times, temps, u0, v0, x_max, n_x, n_t

def run_simulation(config_path):
    # load config
    times, temps, u0, v0, x_max, n_x, n_t = load_sim_config(config_path)

    # pre-compute constants
    x_grid = np.linspace(0, x_max, n_x)

    for T in temps:
        for t_h in times:
            # instantiate & run CN solver
            solver = CNSolver(T, u0, v0, t_h, x_max, n_x, n_t)
            U_record = solver.get_oxygen_profile()
            o_total = U_record[-1]

            # reporting
            report = GenSimReport(x_grid, o_total, t_h, T, "sim_output_const_temp")
            report.plot_overview()
            report.plot_suppression_factor()
            report.plot_suppression_factor_comparison()

            # Test the simulation results against the Ciovati model
            test_oxygen_profile(
                x_grid, t_h, T, o_total, u0, v0,
                output_dir=f"test/bake_{T:.0f}_h_{t_h:.1f}"
            )

            print(f"Done: T={T-273.15:.1f}°C, t={t_h:.1f}h")

    print("All sims complete.")



def main():
    parser = argparse.ArgumentParser(
        description="Run Nb heat-treatment simulation sweep from a config YAML."
    )
    # Optional flag
    parser.add_argument(
        '-c', '--config', dest='config', metavar='CONFIG',
        help="Config file name (in config/) or full path; overrides positional"
    )
    # Positional config argument
    parser.add_argument(
        'pos_config', nargs='?', metavar='CONFIG',
        help="Config file name (in config/) or full path; default: const_temp_sim_config.yml"
    )
    args = parser.parse_args()

    # Choose config: -c overrides positional; fallback to default
    config_name = args.config or args.pos_config or "const_temp_sim_config.yml"
    # Append .yml if missing
    if not os.path.splitext(config_name)[1]:
        config_name += ".yml"

    # Resolve full path: direct if exists, else in config/
    if os.path.isfile(config_name):
        config_path = config_name
    else:
        config_path = os.path.join("config", config_name)

    run_simulation(config_path)


if __name__ == "__main__":
    main()
