import numpy as np
import yaml
import argparse
import os

from solvers.cn_solver import CNSolver
from utils.gen_sim_report import GenSimReport
from utils.gen_temp_profile import gen_temp_profile

from test_sim_heat_treatments import test_oxygen_profile

#from test.test_sim_heat_treatments import test_oxygen_profile


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

def run_simulation(config_path):
    # load config
    start_C, ramp_rate, bake_C_list, times_h, u0, v0, x_max, n_x, n_t = \
        load_sim_config(config_path)

    # pre-compute constants
    start_K = start_C + 273.15
    x_grid  = np.linspace(0, x_max, n_x)

    for bake_C in bake_C_list:
        bake_K = bake_C + 273.15

        for total_h in times_h:
            # generate temperature profile
            time_h, temps_K, t_hold = gen_temp_profile(
                start_K, bake_K, ramp_rate, total_h, n_t,
                exp_b=0.18, exp_c=300.0, tol_K=1.0
            )

            # run solver
            solver = CNSolver(temps_K, u0, v0, total_h, x_max, n_x, n_t)
            U_record = solver.get_oxygen_profile()
            o_total = U_record[-1]

            # reporting
            report = GenSimReport(x_grid, o_total, total_h, bake_K)
            report.plot_overview()
            report.plot_suppression_factor()
            report.plot_suppression_factor_comparison()

            # test against Ciovati model
            test_oxygen_profile(
                x_grid, total_h, bake_K, o_total, u0, v0,
                output_dir=f"test/bake_{bake_C:.0f}_h_{total_h:.1f}"
            )

            print(
                f"Done: bake={bake_C:.0f}°C, total_time={total_h:.1f}h, "
                f"stability={solver.stability}"
            )

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
        help="Config file name (in config/) or full path; default: sim_config.yml"
    )
    args = parser.parse_args()

    # Choose config: -c overrides positional; fallback to default
    config_name = args.config or args.pos_config or "sim_config.yml"
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
