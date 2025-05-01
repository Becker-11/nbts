import numpy as np
import yaml
import argparse
import os

from simulation.cn_solver import CNSolver
from simulation.cn_solver_const_temp import CNSolver as CNSolverConstTemp
from simulation.gen_sim_report import GenSimReport
from simulation.gen_temp_profile import gen_temp_profile
from config.sim_config import load_sim_config

from test_sim_heat_treatments import test_oxygen_profile

#from test.test_sim_heat_treatments import test_oxygen_profile

def run_simulation(cfg, sim_const_temp: bool = False):
    """
    Run the Nb heat-treatment simulation sweep.

    Parameters:
        cfg: Loaded simulation configuration
        sim_const_temp: If True, use constant temperature profile
    """
    # build arrays for sweep
    times_h    = np.arange(
        cfg.time.start_h,
        cfg.time.stop_h  + cfg.time.step_h,
        cfg.time.step_h
    )
    bake_C_list = np.arange(
        cfg.temperature.start_C,
        cfg.temperature.stop_C  + cfg.temperature.step_C,
        cfg.temperature.step_C
    )

    # pre-compute constants
    start_K = cfg.temp_profile.start_C + 273.15
    x_grid  = np.linspace(0, cfg.grid.x_max_nm, cfg.grid.n_x)

    for bake_C in bake_C_list:
        bake_K = bake_C + 273.15

        for total_h in times_h:
            # choose solver based on const flag
            if sim_const_temp:
                # constant temperature solver
                solver = CNSolverConstTemp(
                    bake_K,
                    cfg.initial.u0,
                    cfg.initial.v0,
                    total_h,
                    cfg.grid.x_max_nm,
                    cfg.grid.n_x,
                    cfg.grid.n_t
                )
            else:
                # generate ramp-hold-cool profile
                time_h, temps_K, t_hold = gen_temp_profile(
                    start_K,
                    bake_K,
                    cfg.temp_profile.ramp_rate_C_per_min,
                    total_h,
                    cfg.grid.n_t,
                    exp_b  = cfg.temp_profile.exp_b,
                    exp_c  = cfg.temp_profile.exp_c,
                    tol_K  = cfg.temp_profile.tol_K
                )
                solver = CNSolver(
                    temps_K,
                    cfg.initial.u0,
                    cfg.initial.v0,
                    total_h,
                    cfg.grid.x_max_nm,
                    cfg.grid.n_x,
                    cfg.grid.n_t
                )

            # simulate
            U_record = solver.get_oxygen_profile()
            o_total   = U_record[-1]

            # reporting
            output_dir = cfg.output.directory
            if sim_const_temp:
                output_dir = f"{output_dir}_const_temp"
            report = GenSimReport(
                x_grid,
                o_total,
                total_h,
                bake_K,
                output_dir
            )
            report.plot_overview()
            report.plot_suppression_factor()
            report.plot_suppression_factor_comparison()

            # test against Ciovati model
            test_dir = os.path.join(
                "test_output",
                f"bake_{bake_C:.0f}_h_{total_h:.1f}"
            )
            test_oxygen_profile(
                x_grid,
                total_h,
                bake_C,
                o_total,
                cfg.initial.u0,
                cfg.initial.v0,
                output_dir = test_dir
            )

            print(
                f"Done: bake={bake_C:.0f}°C, total_time={total_h:.1f}h, "
                f"stability={solver.stability}, const_temp={sim_const_temp}"
            )

    print("All sims complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Run Nb heat-treatment simulation sweep from a config YAML."
    )
    parser.add_argument(
        "-c", "--config",
        dest="config",
        metavar="CONFIG",
        help="Config file name or full path; overrides positional"
    )
    parser.add_argument(
        "pos_config",
        nargs="?",
        metavar="CONFIG",
        help="Config file name or full path; default: sim_config.yml"
    )
    parser.add_argument(
        "--const",
        dest="sim_const_temp",
        action="store_true",
        help="Use constant temperature profile"
    )
    args = parser.parse_args()

    # determine config path
    config_name = args.config or args.pos_config or "sim_config.yml"
    if not os.path.splitext(config_name)[1]:
        config_name += ".yml"
    if os.path.isfile(config_name):
        config_path = config_name
    else:
        config_path = os.path.join("config", config_name)

    # load and run
    cfg = load_sim_config(config_path)
    run_simulation(cfg, sim_const_temp=args.sim_const_temp)


if __name__ == "__main__":
    main()

