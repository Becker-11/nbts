#!/usr/bin/env python3
"""
temp_profile.py

Defines temperature-profile generators for SRF bake simulations.

Classes
-------
- BaseTempProfile : Abstract interface
- ConstantProfile : Flat bake at a single temperature
- ThreePhaseProfile: Ramp -> Hold -> Exponential cool

Usage
-----
From your simulation:
    profile = ThreePhaseProfile(cfg, start_K, bake_K, total_h)
    time_h, temps_K, t_hold_min = profile.generate()

Run as a script to see example plots:
    $ python temp_profile.py
"""

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from types import SimpleNamespace


class BaseTempProfile(ABC):
    """Abstract interface for temperature profiles."""

    def __init__(self,
                 cfg: SimpleNamespace,
                 start_K: float,
                 bake_K: float,
                 total_h: float):
        """
        Parameters
        ----------
        cfg : object
            Must have cfg.grid.n_t (number of time points) and, for
            three-phase, cfg.temp_profile.ramp_rate_C_per_min,
            .exp_b, .exp_c, .tol_K.
        start_K : float
            Starting temperature in K.
        bake_K : float
            Peak (bake) temperature in K.
        total_h : float
            Hold duration at bake_K in hours.
        """
        self.cfg     = cfg
        self.start_K = start_K
        self.bake_K  = bake_K
        self.total_h = total_h
        self.n_t     = cfg.grid.n_t  # total number of output steps

    @abstractmethod
    def generate(self) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Produce the profile.

        Returns
        -------
        time_h : np.ndarray, shape (n_t,)
            Time axis, in hours.
        temps_K : np.ndarray, shape (n_t,)
            Temperature at each time point, in K.
        t_hold_min : float
            Hold-time in minutes (bake phase).
        """
        ...


class ConstantProfile(BaseTempProfile):
    """Keeps temperature flat at bake_K for the entire run."""

    def generate(self):
        # Uniform time grid from 0 -> total_h hours
        time_h     = np.linspace(0.0, self.total_h, self.n_t)
        # Flat-line at bake_K
        temps_K    = np.full(self.n_t, self.bake_K, dtype=np.double)
        # Everything is "hold" here
        t_hold_h   = self.total_h
        return time_h, temps_K, t_hold_h


class TimeDepProfile(BaseTempProfile):
    """
    Generate a three-phase bake temperature profile in Kelvin, driven by config.

    Phases:
      1. HEATING: linear ramp from start_K to bake_K at ramp_rate (K/min).
      2. HOLD: constant at bake_K for total_h hours.
      3. COOLING: exponential decay back toward exp_c.

    Parameters
    ----------
    cfg : SimConfig-like object
        Must have attributes:
          cfg.temp_profile.ramp_rate_C_per_min  Ramp rate in K/min
          cfg.grid.n_t                          Number of time steps
          cfg.temp_profile.exp_b               Cooling rate constant (1/h)
          cfg.temp_profile.exp_c               Cooling asymptote (K)
          cfg.temp_profile.tol_K               Cooling tolerance (K)
    start_K : float
        Starting temperature in Kelvin.
    bake_K : float
        Peak (bake) temperature in Kelvin.
    total_h : float
        Duration to hold at bake_K, in hours.

    Returns
    -------
    time_h : np.ndarray, shape (n_t,)
        Time axis in hours.
    temps_K : np.ndarray, shape (n_t,)
        Temperature profile in Kelvin.
    t_hold_hrs : float
        Hold time in hours.
    """

    def generate(self):
        # Unpack config
        ramp_rate = self.cfg.temp_profile.ramp_rate_C_per_min  # K/min
        exp_b     = self.cfg.temp_profile.exp_b                # 1/h
        exp_c     = self.cfg.temp_profile.exp_c                # asymptote K
        tol_K     = self.cfg.temp_profile.tol_K                # tolerance K

        # dynamic amplitude for cooling: a_dyn * exp(-b*t) + exp_c
        a_dyn = self.bake_K - exp_c
        if a_dyn <= 0:
            raise ValueError(
                f"Bake_K ({self.bake_K}) must exceed exp_c ({exp_c})"
            )

        # convert cooling rate to per-minute
        b_per_min = exp_b / 60.0

        # compute durations (in minutes)
        t_heat = (self.bake_K - self.start_K) / ramp_rate
        t_hold = self.total_h * 60.0
        t_cool = (1.0 / b_per_min) * np.log(a_dyn / tol_K)
        total_min = t_heat + t_hold + t_cool

        # uniform time grid in minutes
        t = np.linspace(0.0, total_min, self.n_t)

        # piecewise definition
        temps_K = np.piecewise(
            t,
            [t <= t_heat,
             (t > t_heat) & (t <= t_heat + t_hold),
             t > t_heat + t_hold],
            [
                # ramp up
                lambda tau: self.start_K + ramp_rate * tau,
                # hold
                lambda tau: self.bake_K,
                # exponential cooldown
                lambda tau: a_dyn * np.exp(
                    -b_per_min * (tau - t_heat - t_hold)
                ) + exp_c
            ]
        )

        # convert minutes -> hours for the x-axis
        time_h     = t / 60.0
        return time_h, temps_K, t_hold
    


class TwoStepProfile(BaseTempProfile):
    """
    Two‐step bake profile:
    1. Ramp   from start_K → t1_K
    2. Hold   at t1_K for bake1_h hours
    3. Ramp   from t1_K → t2_K
    4. Hold   at t2_K for bake2_h hours
    5. Cool   exponentially toward exp_c
    """
    def generate(self):
        # Unpack shared config
        ramp_rate = self.cfg.temp_profile.ramp_rate_C_per_min  # K/min
        exp_b     = self.cfg.temp_profile.exp_b                # 1/h
        exp_c     = self.cfg.temp_profile.exp_c                # asymptote K
        tol_K     = self.cfg.temp_profile.tol_K                # tolerance K

        # Unpack two‐step params
        ts       = self.cfg.temp_profile.two_step
        t1_K     = ts.t1_C    + 273.15  # first‐step peak
        bake1_h  = ts.bake1_h           # hold time at t1_K (h)
        t2_K     = ts.t2_C    + 273.15  # second‐step peak
        bake2_h  = ts.bake2_h           # hold time at t2_K (h)

        # dynamic amplitude for final cooldown
        a_dyn = t2_K - exp_c
        if a_dyn <= 0:
            raise ValueError(f"T2 ({t2_K}) must exceed exp_c ({exp_c})")

        # convert cooling rate to per‐minute
        b_per_min = exp_b / 60.0

        # compute durations (minutes)
        t_ramp1 = (t1_K - self.start_K) / ramp_rate
        t_hold1 = bake1_h * 60.0
        t_ramp2 = (t2_K - t1_K) / ramp_rate
        t_hold2 = bake2_h * 60.0
        t_cool  = (1.0 / b_per_min) * np.log(a_dyn / tol_K)
        total_min = t_ramp1 + t_hold1 + t_ramp2 + t_hold2 + t_cool

        # uniform time grid in minutes
        t = np.linspace(0.0, total_min, self.n_t)

        # piecewise definition
        temps_K = np.piecewise(
            t,
            [
                t <= t_ramp1,
                (t > t_ramp1) &
                (t <= t_ramp1 + t_hold1),
                (t > t_ramp1 + t_hold1) &
                (t <= t_ramp1 + t_hold1 + t_ramp2),
                (t > t_ramp1 + t_hold1 + t_ramp2) &
                (t <= t_ramp1 + t_hold1 + t_ramp2 + t_hold2),
                t > t_ramp1 + t_hold1 + t_ramp2 + t_hold2
            ],
            [
                # 1) ramp up to T1
                lambda τ: self.start_K + ramp_rate * τ,
                # 2) hold at T1
                lambda τ: t1_K,
                # 3) ramp up to T2
                lambda τ: t1_K + ramp_rate * (τ - t_ramp1 - t_hold1),
                # 4) hold at T2
                lambda τ: t2_K,
                # 5) exponential cooldown
                lambda τ: a_dyn * np.exp(
                    -b_per_min * (τ - t_ramp1 - t_hold1 - t_ramp2 - t_hold2)
                ) + exp_c
            ]
        )

        # convert minutes → hours for the x‐axis
        time_h = t / 60.0
        # return exactly like ThreePhaseProfile: holds combined in minutes
        t_hold_total = t_hold1 + t_hold2
        return time_h, temps_K, t_hold_total



def main():
    """Quick visual sanity-check of constant, three-phase, and two-step profiles."""
    # dummy config stub
    cfg = SimpleNamespace(
        temp_profile=SimpleNamespace(
            ramp_rate_C_per_min=2.0,   # 2 K/min
            exp_b=0.18,                # 0.18 1/h
            exp_c=300.0,               # 300 K asymptote
            tol_K=1.0,                 # stop when within 1 K
            two_step=SimpleNamespace(
                t1_C=80.0,    # first-step peak temp (°C)
                bake1_h=24.0, # hold at t1 for 24 h
                t2_C=120.0,   # second-step peak temp (°C)
                bake2_h=48.0  # hold at t2 for 48 h
            )
        ),
        grid=SimpleNamespace(
            n_t=2001                  # 2001 time points
        )
    )

    # example parameters (°C → K)
    start_C, bake_C, total_h = 20.0, 120.0, 48.0
    start_K = start_C + 273.15
    bake_K  = bake_C  + 273.15

    # instantiate profiles
    three = TimeDepProfile(cfg, start_K, bake_K, total_h)
    const = ConstantProfile(cfg, start_K, bake_K, total_h)
    two   = TwoStepProfile(   cfg, start_K, bake_K, total_h)

    # generate
    t3, T3, _   = three.generate()
    tc, Tc, _   = const.generate()
    t2s, T2s, _ = two.generate()

    # plot them all
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(t3,  T3,  label='Three-Phase Ramp→Hold→Cool')
    ax.plot(tc,  Tc,  '--', label='Constant @ Bake Temp')
    ax.plot(t2s, T2s, '-.', label='Two-Step Ramp→Hold1→Ramp→Hold2→Cool')

    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Temperature (K)')
    ax2 = ax.secondary_yaxis(
        'right',
        functions=(lambda x: x - 273.15, lambda x: x + 273.15)
    )
    ax2.set_ylabel('Temperature (°C)')

    ax.legend(loc='best')
    plt.title('Temperature Profile Examples')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
