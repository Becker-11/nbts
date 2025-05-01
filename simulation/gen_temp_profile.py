import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace


def gen_temp_profile(cfg, start_K, bake_K, total_h):
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
    t_hold_min : float
        Hold time in minutes.
    """
    # Unpack profile parameters from config
    ramp_rate = cfg.temp_profile.ramp_rate_C_per_min  # K per minute
    num_steps = cfg.grid.n_t                          # number of output points
    exp_b     = cfg.temp_profile.exp_b                # 1/h
    exp_c     = cfg.temp_profile.exp_c                # K
    tol_K     = cfg.temp_profile.tol_K                # K tolerance

    # Compute cooling amplitude
    a_dyn = bake_K - exp_c
    if a_dyn <= 0:
        raise ValueError("Bake_K must exceed exp_c for positive cooling amplitude.")

    # Convert exp_b from 1/hour to 1/minute
    b_per_min = exp_b / 60.0

    # Calculate phase durations
    t_heat = (bake_K - start_K) / ramp_rate        # heating duration in minutes
    t_hold = total_h * 60.0                       # hold duration in minutes
    t_cool = (1.0 / b_per_min) * np.log(a_dyn / tol_K)  # cooling duration in minutes

    # Total cycle time in minutes
    total_min = t_heat + t_hold + t_cool

    # Build uniform time grid (minutes)
    t = np.linspace(0.0, total_min, num_steps)

    # Construct piecewise profile
    temps_K = np.piecewise(
        t,
        [t <= t_heat,
         (t > t_heat) & (t <= t_heat + t_hold),
         t > t_heat + t_hold],
        [
            # heating phase
            lambda tau: start_K + ramp_rate * tau,
            # constant hold phase
            lambda tau: bake_K,
            # cooling phase (shift time origin)
            lambda tau: a_dyn * np.exp(-b_per_min * (tau - t_heat - t_hold)) + exp_c
        ]
    )

    # Convert time axis to hours
    time_h = t / 60.0

    return time_h, temps_K, t_hold

def main():
    """
    Example usage and quick test of gen_temp_profile.
    Creates a dummy config, generates the profile, prints summary, and plots it.
    """
    # ---- Example config stub ----
    cfg = SimpleNamespace(
        temp_profile=SimpleNamespace(
            ramp_rate_C_per_min=1.0,
            exp_b=0.18,
            exp_c=300.0,
            tol_K=1.0
        ),
        grid=SimpleNamespace(
            n_t=3001
        )
    )

    # Example parameters
    start_C = 25.0     # °C
    bake_C  = 120.0    # °C
    total_h = 48.0     # hours

    # Convert to Kelvin
    start_K = start_C + 273.15
    bake_K  = bake_C  + 273.15

    # Generate profile
    time_h, temps_K, t_hold_min = gen_temp_profile(cfg, start_K, bake_K, total_h)

    print(f"Profile generated: {len(time_h)} points; hold time = {t_hold_min/60:.1f} hrs")

    # Plot with dual y-axes
    fig, ax_k = plt.subplots()
    ax_k.plot(time_h, temps_K, label='T (K)')
    ax_k.axhline(bake_K, color='gray', linestyle='--', label=f'Bake {bake_C:.0f}°C')
    ax_k.set_xlabel('Time (h)')
    ax_k.set_ylabel('Temperature (K)', color='tab:orange')
    ax_k.tick_params(axis='y', labelcolor='tab:orange')

    # secondary °C axis
    def K_to_C(x): return x - 273.15
    def C_to_K(x): return x + 273.15
    ax_c = ax_k.secondary_yaxis('right', functions=(K_to_C, C_to_K))
    ax_c.set_ylabel('Temperature (°C)', color='tab:blue')
    ax_c.tick_params(axis='y', labelcolor='tab:blue')

    # legend and title
    ax_k.legend(loc='upper right')
    plt.title(f"Ramp→Bake→Cool Profile, hold {total_h:.1f}h")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()