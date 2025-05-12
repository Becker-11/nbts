import numpy as np
import argparse
import os
import time

from simulation.cn_solver import CNSolver
from simulation.sim_report import GenSimReport
from simulation.temp_profile import TimeDepProfile, TwoStepProfile, ConstantProfile
from config.sim_config import load_sim_config
from simulation.ciovati_model import CiovatiModel

from test_sim_heat_treatments import test_oxygen_profile

def run_simulation(cfg, profile: str = "time_dep", reoxidize: bool = False):
    """
    Run the Nb heat-treatment simulation sweep.

    Parameters:
        cfg: Loaded simulation configuration
        profile: Which temperature profile to use:
                 "const", "time_dep", or "two_step"
    """
    # build arrays for sweep
    times_h = np.arange(
        cfg.time.start_h,
        cfg.time.stop_h + cfg.time.step_h,
        cfg.time.step_h
    )
    bake_C_list = np.arange(
        cfg.temperature.start_C,
        cfg.temperature.stop_C + cfg.temperature.step_C,
        cfg.temperature.step_C
    )

    # pre-compute constants
    start_K = cfg.temp_profile.start_C + 273.15
    x_grid  = np.linspace(0, cfg.grid.x_max_nm, cfg.grid.n_x)

    # initialize Ciovati model
    civ_model = CiovatiModel(cfg.ciovati)

    # select profile class and output suffix
    match profile:
        case "const":
            ProfileClass = ConstantProfile
            suffix = "_const_temp"
        case "time_dep":
            ProfileClass = TimeDepProfile
            suffix = ""
        case "two_step":
            ProfileClass = TwoStepProfile
            suffix = "_two_step"
        case _:
            raise ValueError(f"Unknown profile: {profile!r}")

    base_output = cfg.output.directory

    for bake_C in bake_C_list:
        bake_K = bake_C + 273.15

        for t_h in times_h:
            start = time.perf_counter()

            # set output directory based on profile
            output_dir = f"{base_output}{suffix}"

            # instantiate and run profile
            # TODO: fix t_hold to total_hours and use in CNsolver
            time_h, temps_K, total_hours = ProfileClass(cfg, start_K, bake_K, t_h).generate()

            print(f"Running {profile} profile @ {bake_C}°C, {total_hours:.2f}h")
            solver = CNSolver(cfg, temps_K, total_hours, civ_model)
            U_record = solver.get_oxygen_profile()
            if reoxidize:
                solver = CNSolver(cfg, temps_K, total_hours, civ_model, U_initial=U_record[-1])
                U_record = solver.get_oxygen_profile()
            o_total   = U_record[-1]

            # generate reports
            report = GenSimReport(
                cfg,
                x_grid,
                o_total,
                total_hours,
                bake_K,
                output_dir
            )
            report.plot_overview()
            report.plot_suppression_factor()
            report.plot_suppression_factor_comparison()

            # run Ciovati model comparison
            test_dir = os.path.join(
                "test_output",
                f"bake_{bake_C:.0f}_h_{total_hours:.1f}"
            )
            test_oxygen_profile(
                cfg,
                x_grid,
                total_hours,
                bake_K,
                o_total,
                output_dir=test_dir
            )
            end = time.perf_counter()

            print(
                f"Done: bake={bake_C:.0f}°C, total_time={total_hours:.1f}h, "
                f"profile={profile}, "
                f"Completed in {end - start:.2f}s"
            )

            #print(f"stability={solver.stability} r={solver.r}")

    print(f"All sims complete. Plots and Data can be found in: {output_dir}")



def main():

    """
        Run Nb heat-treatment simulation sweep from a config YAML.

        Usage:
            sim [-c CONFIG] [-p PROFILE] [CONFIG_FILE]
            sim reoxidize [-c CONFIG] [-p PROFILE] [CONFIG_FILE]

        Positional arguments:
            CONFIG_FILE            Config file name or full path (default: sim_config.yml)

        Optional arguments:
            -h, --help             show this help message and exit
            -c CONFIG, --config CONFIG
                                Config file name or full path; overrides positional
            -p PROFILE, --profile PROFILE
                                Temperature profile to use: const, time_dep, or two_step

        Sub-commands:
            reoxidize              Run in reoxidation mode (sets reoxidize=True)
    """
    
    parser = argparse.ArgumentParser(
        description="Run Nb heat-treatment simulation sweep from a config YAML."
    )
    # these apply to both commands:
    parser.add_argument(
        "-c", "--config",
        dest="config",
        metavar="CONFIG",
        help="Config file name or full path; overrides positional"
    )
    parser.add_argument(
        "-p", "--profile",
        dest="profile",
        choices=["const", "time_dep", "two_step"],
        default="time_dep",
        help=(
            "Temperature profile to use: "
            "`const`, `time_dep`, or `two_step`"
        )
    )
    parser.add_argument(
        "pos_config",
        nargs="?",
        metavar="CONFIG",
        help="Positional config file; default: sim_config.yml"
    )

    # sub-commands: 
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("reoxidize",
                          help="Run in reoxidation mode (sets reoxidize=True)")

    args = parser.parse_args()

    # figure out config path 
    config_name = args.config or args.pos_config or "sim_config.yml"
    if not os.path.splitext(config_name)[1]:
        config_name += ".yml"
    if os.path.isfile(config_name):
        config_path = config_name
    else:
        config_path = os.path.join("config", config_name)

    # load
    cfg = load_sim_config(config_path)

    # detect reoxidize
    reoxidize = (args.command == "reoxidize")

    # dispatch
    run_simulation(
        cfg,
        profile=args.profile,
        reoxidize=reoxidize
    )

if __name__ == "__main__":
    main()

