import numpy as np
import matplotlib.pyplot as plt

def gen_temp_profile(
    start_K, bake_K, ramp_rate_K_per_min, total_time_hours, num_steps,
    exp_b=0.18, exp_c=300.0, tol_K=1.0
):
    """
    Generate a three‐phase bake profile in Kelvin:
      1. HEATING: linear ramp from start_K → bake_K at ramp_rate_K_per_min (K/min).
      2. HOLD: constant at bake_K for any leftover time.
      3. COOLING: exp decay a·exp(–b t) + exp_c, down to within tol_K of exp_c.

    Parameters
    ----------
    start_K : float
        Starting temperature (K).
    bake_K : float
        Bake (peak) temperature (K).
    ramp_rate_K_per_min : float
        Ramp rate in K per minute.
    total_time_hours : float
        Total cycle time (h).
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
    """
    # 1) Compute cooling amplitude so T(0)=bake_K
    a_dyn = bake_K - exp_c
    if a_dyn <= 0:
        raise ValueError("Bake_K must exceed exp_c for a positive amplitude.")

    # 2) Convert exp_b from 1/h → 1/min
    b_per_min = exp_b / 60.0

    # 3) Phase durations in minutes
    t_heat = (bake_K - start_K) / ramp_rate_K_per_min
    # time to cool until exp_c + tol_K
    t_cool = (1.0 / b_per_min) * np.log(a_dyn / tol_K)
    total_min = total_time_hours * 60.0

    # 4) Any leftover is hold time
    t_hold = total_min - t_heat - t_cool
    if t_hold < 0:
        # Not enough time for a full tol_K cooldown → skip hold, shorten cooling:
        t_hold = 0.0
        t_cool = total_min - t_heat
        if t_cool <= 0:
            raise ValueError("Total time too short to finish heating.")

    # 5) Build uniform time grid
    t = np.linspace(0.0, total_min, num_steps)

    # 6) Piecewise profile
    temps_K = np.piecewise(
        t,
        [t <= t_heat,
         (t > t_heat) & (t <= t_heat + t_hold),
         t > t_heat + t_hold],
        [
            # heating
            lambda τ: start_K + ramp_rate_K_per_min * τ,
            # hold
            lambda τ: bake_K,
            # cooling (shift τ so 0 at start of cooling)
            lambda τ: a_dyn * np.exp(-b_per_min * (τ - t_heat - t_hold)) + exp_c
        ]
    )

    time_h = t / 60.0
    return time_h, temps_K, t_hold


def main():
    # ---- user inputs ----
    start_C = 25.0            # °C
    bake_C  = 120.0           # °C
    ramp_rate = 1.0           # °C/min  (same as K/min)
    total_hours = 48.0        # h
    steps = 3001              # resolution

    # convert to Kelvin
    start_K = start_C + 273.15
    bake_K  = bake_C  + 273.15

    # generate profile
    time_h, temps_K, t_hold = gen_temp_profile(
        start_K, bake_K, ramp_rate, total_hours, steps,
        exp_b=0.18, exp_c=300.0, tol_K=1.0
    )
    print(f"Profile: {len(temps_K)} points over {total_hours} h "
          f"(heat {time_h[np.argmax(temps_K)]*60:.1f} min, "
          f"hold {max(0, (total_hours*60 - ((bake_K-start_K)/ramp_rate) - ((1/ (0.18/60))*np.log((bake_K-300.0)/1.0)) )):.1f} min).")

    # plot
    fig, ax_k = plt.subplots()
    ax_k.plot(time_h, temps_K, color='tab:orange', label='T (K)')
    ax_k.set_xlabel('Time (h)')
    ax_k.set_ylabel('Temperature (K)', color='tab:orange')
    ax_k.tick_params(axis='y', labelcolor='tab:orange')
    ax_k.axhline(bake_K, color='gray', ls='--', label=f'Bake {bake_C:.0f}°C')

    # secondary °C axis
    def K_to_C(x): return x - 273.15
    def C_to_K(x): return x + 273.15
    ax_c = ax_k.secondary_yaxis('right', functions=(K_to_C, C_to_K))
    ax_c.set_ylabel('Temperature (°C)')
    ax_c.tick_params(axis='y', labelcolor='tab:blue')

    # legend
    handles, labels = ax_k.get_legend_handles_labels()
    ax_k.legend(handles, labels, loc='upper right')

    plt.title(f"Ramp → {bake_C:.0f}°C, hold {t_hold:.1f} min, cool tol {1.0:.1f} K")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
