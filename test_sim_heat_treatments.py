import os
import matplotlib.pyplot as plt
from simulation.ciovati_model import CiovatiModel


def test_oxygen_profile(
    cfg,
    x_grid,
    total_h: float,
    bake_K: float,
    o_total,
    output_dir: str = None
):
    """
    Plot simulation results against Ciovati model predictions and save to output_dir.

    Parameters:
        cfg (SimConfig): Configuration object with model parameters and output directory.
        x_grid (array-like): Spatial grid points (nm).
        total_h (float): Simulation time in hours.
        bake_K (float): Bake temperature in Kelvin.
        o_total (array-like): Simulated oxygen concentration profile.
        output_dir (str, optional): Directory to save the plot. Defaults to cfg.output.directory + '/test'.
    """
    # Determine output directory
    base_dir = cfg.output.directory
    out_dir = output_dir or os.path.join(base_dir, 'test')
    os.makedirs(out_dir, exist_ok=True)

    # Instantiate Ciovati model with parameters from config
    model = CiovatiModel(cfg.ciovati)

    # Compute model predictions
    time_sec = total_h * 3600.0
    c_model = [model.c(x, time_sec, bake_K) for x in x_grid]

    # Plot both profiles
    fig, ax = plt.subplots()
    ax.plot(x_grid, o_total, '-', label='Simulation')
    ax.plot(x_grid, c_model, '-', label='Ciovati Model')
    ax.set_xlabel('Depth (nm)')
    ax.set_ylabel('Oxygen Concentration (at.%)')
    ax.set_title(f'Oxygen Profile at {total_h:.1f} h and {bake_K - 273.15:.1f} K')
    ax.set_xlim(0, 150)
    ax.legend()

    # Save plot
    filename = os.path.join(out_dir, 'oxygen_profile_comparison.pdf')
    fig.savefig(filename)
    print(f"Saved plot to {filename}")


def main():
    pass


if __name__ == '__main__':
    main()
