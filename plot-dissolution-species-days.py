import numpy as np
import matplotlib.pyplot as plt
from dissolution_species import c_Nb2O5, c_NbO2, c_NbO, c_O


# initial concentrations
c_Nb2O5_0 = 100.0
c_NbO2_0 = 0.0
c_NbO_0 = 0.0
c_O_0 = 0.0

# temperature
T_celsius = 120.0
T = T_celsius + 273.15

# time conversions
s_per_min = 60.0
min_per_h = 60.0
h_per_day = 24.0
s_per_h = s_per_min * min_per_h
s_per_day = s_per_h * h_per_day

# baking time
days = 5e4

# time points
t = np.linspace(0.0, days * s_per_day, 1000)

# figure for plotting
fig, ax = plt.subplots(
    1,
    1,
    figsize=(9.6, 3.2),
    constrained_layout=True,
)

# normalization factor (if desired)
norm = 1.0  # c_Nb2O5_0

# plot for Nb2O5
ax.plot(
    t / s_per_day,
    c_Nb2O5(t, T, c_Nb2O5_0) / norm,
    "-",
    zorder=1,
    label=r"Nb$_{2}$O$_{5}$",
)

# plot for NbO2
ax.plot(
    t / s_per_day,
    c_NbO2(t, T, c_Nb2O5_0, c_NbO2_0) / norm,
    "-",
    zorder=1,
    label=r"NbO$_{2}$",
)

# plot for NbO
ax.plot(
    t / s_per_day,
    c_NbO(t, T, c_Nb2O5_0, c_NbO2_0, c_NbO_0) / norm,
    "-",
    zorder=1,
    label=r"NbO",
)

# plot for O
ax.plot(
    t / s_per_day,
    c_O(t, T, c_Nb2O5_0, c_NbO2_0, c_O_0) / norm,
    "-",
    zorder=1,
    label=r"O",
)

# set the axis limits to sensible values
ax.set_xlim(
    t.min() / s_per_day,
    t.max() / s_per_day,
)
ax.set_ylim(0, None)

# add axis labels
ax.set_xlabel("$t$ (days)")
ax.set_ylabel(f"[X] (at. %)")

# create a legend with simulation details in its title
legend = ax.legend(
    title="Simulation parameters:\n\n"
    + f"$T = {T_celsius:.1f}$ $^\\circ$C\n\n"
    + r"$[\mathrm{Nb}_2\mathrm{O}_5]_0 = "
    + f"{c_Nb2O5_0:.1f}$ at. %\n"
    + r"$[\mathrm{NbO}_2]_0 = "
    + f"{c_NbO2_0:.1f}$ at. %\n"
    + r"$[\mathrm{NbO}]_0 = "
    + f"{c_NbO_0:.1f}$ at. %\n"
    + r"$[\mathrm{O}]_0 = "
    + f"{c_O_0:.1f}$ at. %\n\n"
    + "Surface oxide species:\n",
    frameon=False,
    ncol=1,
    # loc="lower center",
    # loc="center left",
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0),
    fontsize="small",
)

# make the legend title font smaller
legend.get_title().set_fontsize("small")

# optionally save the figure with a filename that reflects the simulation conditions
save_figure = False
if save_figure:
    fig.savefig(
        f"oxide-species-{T_celsius:.0f}C-{t.max() / s_per_day:.1f}days.pdf",
    )

# show the figure
plt.show()
