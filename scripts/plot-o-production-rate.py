import numpy as np
import matplotlib.pyplot as plt
from models.dissolution_species import dc_O_dt


# initial concentrations
c_Nb2O5_0 = 100.0
c_NbO2_0 = 0.0


# time conversions
s_per_min = 60.0
min_per_h = 60.0
h_per_day = 24.0
s_per_h = s_per_min * min_per_h
s_per_day = s_per_h * h_per_day

# time points
# t = np.linspace(0.0, 120.0 * s_per_h, 1000)
t_h = np.logspace(-1.0, 3.0, 1000)
t = t_h * s_per_h

# figure for plotting
fig, ax = plt.subplots(
    1,
    1,
    figsize=(4.8, 6.4),
    constrained_layout=True,
)

# temperature
T_celsius = np.linspace(120.0, 360.0, 9)
T = T_celsius + 273.15

for TT in T:

    # plot for O
    ax.plot(
        t / s_per_h,
        dc_O_dt(t, TT, c_Nb2O5_0, c_NbO2_0),
        "-",
        zorder=1,
        label=f"$T = {TT - 273.15:.1f}$ $^\\circ$C",
    )

# set the axis limits to sensible values
ax.set_xlim(
    # None,
    t.min() / s_per_h,
    t.max() / s_per_h,
)

ax.set_ylim(9e-9, None)
ax.set_yscale("log")

ax.set_xscale("log")

# add axis labels
ax.set_xlabel("$t$ (h)")
ax.set_ylabel(r"$\mathrm{d}[\mathrm{O}] \, / \, \mathrm{d}t$ (at. % s$^{-1}$)")


# create a legend with simulation details in its title
legend = ax.legend(
    title="Simulation parameters:\n\n"
    # + f"$T = {T_celsius:.1f}$ $^\\circ$C\n\n"
    + r"$[\mathrm{Nb}_2\mathrm{O}_5]_0 = "
    + f"{c_Nb2O5_0:.1f}$ at. %\n"
    + r"$[\mathrm{NbO}_2]_0 = "
    + f"{c_NbO2_0:.1f}$ at. %\n\n"
    # + r"$[\mathrm{NbO}]_0 = "
    # + f"{c_NbO_0:.1f}$ at. %\n"
    # + r"$[\mathrm{O}]_0 = "
    # + f"{c_O_0:.1f}$ at. %\n\n"
    + "Temperature $T$:\n",
    frameon=False,
    ncol=3,
    # loc="lower center",
    # loc="center left",
    loc="lower center",
    bbox_to_anchor=(0.5, 1.05),
    fontsize="small",
)

# make the legend title font smaller
legend.get_title().set_fontsize("small")


# optionally save the figure with a filename that reflects the simulation conditions
save_figure = False
if save_figure:
    fig.savefig(
        f"o-prodcution-rate-{t.max() / s_per_h:.0f}h.pdf",
    )

# show the figure
plt.show()
