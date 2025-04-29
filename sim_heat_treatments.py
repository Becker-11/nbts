import numpy as np
import yaml
import math
import argparse

from cn_solver import CNSolver
from gen_sim_report import GenSimReport
from gen_temp_profile import gen_temp_profile



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

    # --- bake‐profile sweep parameters ---
    bp     = cfg["bake_profile"]
    start_C = bp["start_C"]               # °C
    ramp_rate = bp["ramp_rate_C_per_min"] # °C/min == K/min

    # build list of recipe‐times (h) and bake‐temps (°C) to sweep:
    t_cfg = cfg["time"]
    times_h = np.arange(t_cfg["start"], t_cfg["stop"] + t_cfg["step"], t_cfg["step"])

    tmp_cfg = cfg["temperature"]
    bake_C_list = np.arange(tmp_cfg["start_c"],
                            tmp_cfg["stop_c"] + tmp_cfg["step_c"],
                            tmp_cfg["step_c"])

    # initial surface amounts
    u0 = cfg["initial"]["u0"]
    v0 = cfg["initial"]["v0"]

    # spatial & solver resolution
    grid = cfg["grid"]
    x_max = grid["x_max_nm"]
    n_x   = grid["n_x"]
    n_t   = grid["t_steps"]

    return start_C, ramp_rate, bake_C_list, times_h, u0, v0, x_max, n_x, n_t



import argparse
import numpy as np
# … your other imports …

def main(config_path):
    # load everything
    start_C, ramp_rate, bake_C_list, times_h, u0, v0, x_max, n_x, n_t = \
        load_sim_config(config_path)

    # pre‐compute constants
    start_K = start_C + 273.15
    x_grid  = np.linspace(0, x_max, n_x)

    for bake_C in bake_C_list:
        bake_K = bake_C + 273.15

        for total_h in times_h:
            # 1) generate full T(t) profile in Kelvin
            time_h, temps_K, t_hold = gen_temp_profile(
                start_K, bake_K, ramp_rate, total_h, n_t,
                exp_b=0.18, exp_c=300.0, tol_K=1.0
            )

            # 2) run your CN‐solver *with* that profile
            solver = CNSolver(temps_K, time_h, u0, v0, x_max, n_x, n_t)
            U_record = solver.get_oxygen_profile()
            o_total = U_record[-1]

            # 3) reporting
            report = GenSimReport(x_grid, o_total, total_h, bake_K)
            report.plot_overview()
            report.plot_suppression_factor()
            report.plot_suppression_factor_comparison()

            print(
                f"Done: bake={bake_C:.0f}°C, total_time={total_h:.1f}h, "
                f"stability={solver.stability}"
            )

    print("All sims complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Nb heat‐treatment simulation sweep from a config YAML."
    )
    parser.add_argument(
        "-c", "--config",
        metavar="CONFIG",
        default="sim_config.yml",
        help="Path to the simulation config file (YAML)."
    )
    args = parser.parse_args()

    main(args.config)
