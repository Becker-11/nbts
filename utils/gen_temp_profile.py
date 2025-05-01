import numpy as np
import matplotlib.pyplot as plt


def gen_temp_profile(
    start_K, bake_K, ramp_rate_K_per_min, hold_time_hours, num_steps,
    exp_b=0.18, exp_c=300.0, tol_K=1.0
):
    """
    Generate a three-phase bake profile in Kelvin:
      1. HEATING: linear ramp from start_K → bake_K at ramp_rate_K_per_min (K/min).
      2. HOLD: constant at bake_K for hold_time_hours.
      3. COOLING: exponential decay a·exp(–b t) + exp_c, down to within tol_K of exp_c.

    Parameters
    ----------
    start_K : float
        Starting temperature (K).
    bake_K : float
        Bake (peak) temperature (K).
    ramp_rate_K_per_min : float
        Ramp rate in K per minute.
    hold_time_hours : float
        Duration to hold at bake_K (h).
    num_steps : int
        Number of points in the output.
    exp_b : float
        Exponential fit rate (1/h).
    exp_c : float
        Exponential fit asymptote (K).
    tol_K : float
        Stop cooling when within tol_K of exp_c.

    Returns
    -------
    time_h : np.ndarray, shape (num_steps,)
        Time axis in hours.
    temps_K : np.ndarray, shape (num_steps,)
        Temperature profile in Kelvin.
    t_hold_min : float
        Hold time in minutes.
    """
    # amplitude for cooling
    a_dyn = bake_K - exp_c
    if a_dyn <= 0:
        raise ValueError("Bake_K must exceed exp_c for a positive amplitude.")

    # convert exp_b from 1/h -> 1/min
    b_per_min = exp_b / 60.0

    # phase durations in minutes
    t_heat = (bake_K - start_K) / ramp_rate_K_per_min
    t_hold = hold_time_hours * 60.0
    t_cool = (1.0 / b_per_min) * np.log(a_dyn / tol_K)

    # total cycle time in minutes
    total_min = t_heat + t_hold + t_cool

    # uniform time grid
    t = np.linspace(0.0, total_min, num_steps)

    # piecewise profile
    temps_K = np.piecewise(
        t,
        [t <= t_heat,
         (t > t_heat) & (t <= t_heat + t_hold),
         t > t_heat + t_hold],
        [
            lambda tau: start_K + ramp_rate_K_per_min * tau,
            lambda tau: bake_K,
            lambda tau: a_dyn * np.exp(-b_per_min * (tau - t_heat - t_hold)) + exp_c
        ]
    )

    time_h = t / 60.0
    return time_h, temps_K, t_hold


def main():
    # ---- user inputs ----
    start_C = 25.0            # °C
    bake_C  = 120.0           # °C
    ramp_rate = 1.0           # °C/min (same as K/min)
    hold_hours = 48.0         # h  (this is now the hold duration)
    steps = 3001              # resolution

    # convert to Kelvin
    start_K = start_C + 273.15
    bake_K  = bake_C  + 273.15

    # generate profile
    time_h, temps_K, t_hold_min = gen_temp_profile(
        start_K, bake_K, ramp_rate, hold_hours, steps,
        exp_b=0.18, exp_c=300.0, tol_K=1.0
    )
    print(
        f"Profile: {len(temps_K)} points;") 

    # plot
    fig, ax_k = plt.subplots()
    ax_k.plot(time_h, temps_K, label='T (K)')
    ax_k.set_xlabel('Time (h)')
    ax_k.set_ylabel('Temperature (K)')
    ax_k.axhline(bake_K, ls='--', label=f'Bake {bake_C:.0f}°C')

    # secondary °C axis
    def K_to_C(x): return x - 273.15
    def C_to_K(x): return x + 273.15
    ax_k.secondary_yaxis('right', functions=(K_to_C, C_to_K)).set_ylabel('Temperature (°C)')

    ax_k.legend()
    plt.title(f"Ramp→Bake→Cool, hold {hold_hours:.1f}h")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
