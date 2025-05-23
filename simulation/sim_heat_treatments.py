#!/usr/bin/env python3
"""sim_heat_treatments.py

Niobium heat‑treatment simulation driver.

The console‑script **``sim``** is installed when you run
``pip install -e .`` inside the *nbts* repository, so you can launch
simulations with either

    $ sim [-c CONFIG] [-p PROFILE] [CONFIG]
    $ sim reoxidize [-p PROFILE] [CONFIG]

or the equivalent

    $ python sim_heat_treatments.py ...

Each run sweeps bake temperature and hold time for a chosen
temperature‑profile class, generates plots/reports, and archives all
provenance under

    experiments/<YYYY‑MM‑DD_HH‑MM‑SS>_<profile>_<git>

Run‑folder contents
-------------------
* ``sim_config.yml``                     – verbatim YAML provided by the user
* ``run_meta.yml``                       – timestamp, CLI flags, git commit hash
* ``effective_config.yml``               – written *only* if CLI flags override YAML
* ``results_<temp_profile>/``            – plots, CSVs, and derived data
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
from dataclasses import asdict

import yaml  # PyYAML
import numpy as np

from simulation.cn_solver import CNSolver
from simulation.sim_report import GenSimReport
from simulation.temp_profile import ConstantProfile, TimeDepProfile, TwoStepProfile
from config.sim_config import load_sim_config
from simulation.ciovati_model import CiovatiModel
from scripts.simulation_analyzer import CurrentDensityAnalyzer

###############################################################################
# ─── Internal helpers ───────────────────────────────────────────────────────
###############################################################################

def _git_hash(short: bool = True) -> str:
    """Return the current commit hash (short or full)."""
    rev = "HEAD"
    cmd = ["git", "rev-parse"]
    cmd += ["--short", rev] if short else [rev]

    # 1) try current working directory
    try:
        return subprocess.check_output(cmd, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # 2) try the directory containing this file
    try:
        return subprocess.check_output(
            cmd, cwd=Path(__file__).resolve().parent,
            text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""          # couldn’t find a repo



def _save_run_metadata(cfg_path: Path, cli_ns: argparse.Namespace, run_dir: Path, cfg_obj) -> None:
    """Archive YAML config, CLI flags, git commit, and effective YAML (if changed)."""
    run_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(cfg_path, run_dir / "sim_config.yml")

    meta = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "git_commit": _git_hash(),
        "cli_flags": {k: v for k, v in vars(cli_ns).items() if k not in {"pos_config", "config"}},
    }
    (run_dir / "run_meta.yml").write_text(yaml.safe_dump(meta, sort_keys=False))

    try:
        merged_yaml = yaml.safe_dump(asdict(cfg_obj), sort_keys=False)
    except TypeError:
        merged_yaml = yaml.safe_dump(cfg_obj.__dict__, sort_keys=False)

    if merged_yaml != Path(cfg_path).read_text():
        (run_dir / "effective_config.yml").write_text(merged_yaml)

###############################################################################
# ─── Core simulation logic ──────────────────────────────────────────────────
###############################################################################

def run_simulation(cfg, profile: str = "time_dep", reoxidize: bool = False, n_reoxidize: int = 0) -> None:
    """Run the simulation sweep for a given configuration and profile."""

    times_h = np.arange(cfg.time.start_h, cfg.time.stop_h + cfg.time.step_h, cfg.time.step_h)
    bake_C_list = np.arange(
        cfg.temperature.start_C,
        cfg.temperature.stop_C + cfg.temperature.step_C,
        cfg.temperature.step_C,
    )

    start_K = cfg.temp_profile.start_C + 273.15
    x_grid = np.linspace(0, cfg.grid.x_max_nm, cfg.grid.n_x)
    civ_model = CiovatiModel(cfg.ciovati)

    match profile:
        case "const":
            ProfileCls, suffix = ConstantProfile, "_const_temp"
        case "time_dep":
            ProfileCls, suffix = TimeDepProfile, ""
        case "two_step":
            ProfileCls, suffix = TwoStepProfile, "_two_step"
        case _:
            raise ValueError(f"Unknown profile: {profile!r}")

    output_dir = cfg.output.directory
    reox_output_dir = output_dir
    total_simulations = len(times_h) * len(bake_C_list)
    threshold = 400

    for bake_C in bake_C_list:
        bake_K = bake_C + 273.15
        for time_hold in times_h:
            tic = time.perf_counter()

            t_h, temps_K, total_h = ProfileCls(cfg, start_K, bake_K, time_hold).generate()
            
            print(f"Running {profile} profile @ {bake_C:.0f}°C, hold time: {time_hold:.2f}h, total time: {total_h:.2f}h")

            solver = CNSolver(cfg, temps_K, total_h, civ_model)
            U_record = solver.get_oxygen_profile()
            o_total = U_record[-1]

            if reoxidize:
                for n in range(n_reoxidize):
                    pass_dir = f"{reox_output_dir}_reoxidize_{n+1}"
                    report = GenSimReport(cfg, x_grid, o_total, time_hold, bake_K, t_h, temps_K, profile, pass_dir)
                    report.generate()
                    print(f"Re-oxidizing {n+1} pass of {n_reoxidize}...")
                    solver = CNSolver(cfg, temps_K, total_h, civ_model, U_initial=o_total)
                    U_record = solver.get_oxygen_profile()
                    o_total = U_record[-1]
                    print(f"Re-oxidization pass {n+1} complete. output → {pass_dir}")
                output_dir = f"{reox_output_dir}_reoxidized"
            report = GenSimReport(cfg, x_grid, o_total, time_hold, bake_K, t_h, temps_K, profile, output_dir)
            report.generate()


            # ── Completion message with timing ──
            elapsed = time.perf_counter() - tic
            print(f"Done: {profile} profile @ {bake_C:.0f}°C, hold time: {time_hold:.2f}h, total_time={total_h:.2f}h,\n "
                  f"Completed in {elapsed:.2f}s, output → {output_dir}")

    # TODO: choose a better threshold for analysis
    print(f"Total simulations: {total_simulations}")
    if total_simulations > threshold:
        # ── Generate analysis report ──────────────────────────────
        if reoxidize:
            CurrentDensityAnalyzer(output_dir, "reoxidized").run()
            for n in range(n_reoxidize):
                pass_dir = f"{reox_output_dir}_reoxidize_{n+1}"
                CurrentDensityAnalyzer(pass_dir, f"reoxidize_{n+1}").run()
        else:
            CurrentDensityAnalyzer(output_dir).run()

###############################################################################
# ─── Command‑line interface ─────────────────────────────────────────────────
###############################################################################

def main() -> None:
    """Entry point for the *sim* console script."""
    parser = argparse.ArgumentParser(
        description="Run Nb heat‑treatment simulation sweep from a YAML config.",
        prog="sim",
    )

    parser.add_argument("-c", "--config", metavar="CONFIG", help="YAML config file")

    parser.add_argument(
        "-p", "--profile",
        choices=["const", "time_dep", "two_step"],
        default="time_dep",
        help="Temperature profile to use",
    )

    parser.add_argument(
        "-r", "--reoxidize",
        metavar="N",
        type=int,
        nargs="?",
        const=1,
        default=0,
        help="Number of re‑oxidization passes (omit flag for 0)",
    )

    parser.add_argument("pos_config", nargs="?", metavar="CONFIG", help="Positional YAML config")
    args = parser.parse_args()

    # ── Resolve config path ─────────────────────────────────────
    cfg_name = args.config or args.pos_config or "sim_config.yml"
    if not os.path.splitext(cfg_name)[1]:
        cfg_name += ".yml"
    cfg_path = Path(cfg_name) if Path(cfg_name).is_file() else Path("config") / cfg_name
    if not cfg_path.exists():
        parser.error(f"Config file not found: {cfg_path}")

    cfg = load_sim_config(cfg_path)

    # ── Prepare output directory ────────────────────────────────
    stamp = datetime.now().strftime("%Y-%m-%d")
    ghash = _git_hash() or "no-git"
    run_dir = Path("experiments") / f"{stamp}_{args.profile}_{ghash}"
    cfg.output.directory = str(run_dir / "results")

    _save_run_metadata(cfg_path, args, run_dir, cfg)

    # ── Pass the count, plus boolean flag ──────────────────────
    run_simulation(
        cfg,
        profile=args.profile,
        reoxidize=(args.reoxidize > 0),
        n_reoxidize=args.reoxidize,
    )


if __name__ == "__main__":
    main()
