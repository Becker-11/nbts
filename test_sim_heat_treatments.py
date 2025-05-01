import os
import matplotlib.pyplot as plt
from models.ciovati_model import c as ciovati_c

def test_oxygen_profile(x_grid, time, temp, o_total, u0, v0, output_dir='test'):
    """
    Plot simulation results against Ciovati model predictions and save to output_dir.

    Parameters:
        x_grid (array-like): Spatial grid points.
        time (float or array-like): Time or times at which to evaluate the model.
        temp (array-like): Temperature profile corresponding to time.
        o_total (array-like): Simulated oxygen concentration profile at x_grid.
        u0: Initial condition parameter for Ciovati model.
        v0: Initial condition parameter for Ciovati model.
        output_dir (str): Directory to save the plot.
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    #print(f'x={max(x_grid)}, time={time}, temp={temp}, u0={u0}, v0={v0}')
    # Compute model concentrations
    time_sec = time * 3600.0  # Convert hours to seconds
    c_model = [ciovati_c(x, time_sec, temp, u0, v0, 0) for x in x_grid]

    # Plot both profiles
    fig, ax = plt.subplots()
    ax.plot(x_grid, o_total, '-', label='Simulation')
    ax.plot(x_grid, c_model, '-', label='Ciovati Model')
    ax.set_xlabel('Depth')
    ax.set_ylabel('Oxygen Concentration')
    ax.set_title('Oxygen Profile: Simulation vs Ciovati Model')
    ax.set_xlim(0, 150)
    ax.legend()

    fig.tight_layout()

    # Save plot
    filename = os.path.join(output_dir, 'oxygen_profile_comparison.pdf')
    fig.savefig(filename)
    #print(f"Saved plot to {filename}")

    # Optionally display plot
    #plt.show()

def main():
    return

if __name__ == '__main__':
    main()    
