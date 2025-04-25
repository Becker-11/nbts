import numpy as np
from scipy import constants, integrate, interpolate
import matplotlib.pyplot as plt
from model import c, u, v
from quantities import ell, lambda_eff, J, B
from cn_solver import CNSolver
from gle_solver import GLESolver


# 5-panel figure to show the connected quantities
fig, axes = plt.subplots(
    5,
    1,
    sharex=True,
    sharey=False,
    figsize=(4.8, 6.4 + 0.5 * 3.2),
    constrained_layout=True,
)

# array of depth points in nm
# note: the large endpoint is necessary for obtaining accurate solutions to the
#       generalized london equation later on
x = np.linspace(0, 1000, 10001)

# simulation conditions for the low-temperature baking
s_per_min = 60.0
min_per_h = 60.0
s_per_h = s_per_min * min_per_h
t = 12 * s_per_h  # s
T = 273.15 + 120  # K
u_0 = 1000.0  # at. % nm
v_0 = 10.0  # at. % nm
c_0 = 0.01  # at. % nm

# simulate the components of the oxygen diffusion profile using Ciovati's
# simplified model
o_surf = np.array([u(xx, t, T, u_0) for xx in x])
o_inter = np.array([v(xx, t, T, v_0, c_0) for xx in x])
o_total = np.array([c(xx, t, T, u_0, v_0, c_0) for xx in x])

#o_total = oxygen_profile()[-1]

# plot the oxygen diffusion profile
axes[0].plot(
    x,
    o_total,
    "-",
    zorder=2,
    # color="C2",
    label=r"Total: $c(x) = u(x) + v(x)$",
)
axes[0].set_ylabel(r"$[\mathrm{O}]$ (at. %)")

# annotate the plot with the simulation conditions
axes[0].text(
    0.6,
    0.9,
    f"$t = {t / 3600.0:.1f}$ h\n"
    + f"$T = {T - 273.15:.1f}$ $^\\circ$C\n"
    + f"$u_0 = {u_0:.1f}$ at. % nm\n"
    + f"$v_0 = {v_0:.1f}$ at. % nm\n"
    + f"$c_0 = {c_0:.2f}$ at. %",
    ha="left",
    va="top",
    fontsize="small",
    transform=axes[0].transAxes,
)
axes[0].text(
    0.55,
    0.9,
    "Nb baking conditions:",
    ha="right",
    va="top",
    fontsize="small",
    transform=axes[0].transAxes,
)

# calculate the electron mean-free-path from the diffusion profile
ell = ell(o_total)

# calculate the effective penetration depth using the electron mean-free-path
lambda_eff = lambda_eff(ell)

# plot the depth-dependent mean-free-path
axes[1].plot(
    x,
    ell,
    "-",
    zorder=1,
)
axes[1].axhline(ell.min(), linestyle=":", color="C1", zorder=0)
axes[1].axhline(ell.max(), linestyle=":", color="C2", zorder=0)
axes[1].set_ylabel(r"$\ell$ (nm)")
axes[1].set_ylim(0, None)


# plot the depth-dependent penetration depth
axes[2].plot(
    x,
    lambda_eff,
    "-",
    zorder=1,
)
axes[2].axhline(lambda_eff.min(), linestyle=":", color="C2", zorder=0)
axes[2].axhline(lambda_eff.max(), linestyle=":", color="C1", zorder=0)
axes[2].set_ylabel(r"$\lambda_{\mathrm{eff.}}$ (nm)")


# initialize the GLE solver using the depth/penetration depth sampling points
gle = GLESolver(
    x,
    lambda_eff,
)

# common arguments for the screening/currrent density profiles
args = (100.0, 0.0, 0.0)

# plot the Meissner screening profile
axes[3].plot(
    x,
    gle.screening_profile(x, *args),
    "-",
    zorder=1,
)
axes[3].plot(
    x,
    B(x, args[0], lambda_eff.max()),
    ":",
    color="C1",
    zorder=0,
)
axes[3].plot(
    x,
    B(x, args[0], lambda_eff.min()),
    ":",
    color="C2",
    zorder=0,
)
axes[3].set_ylim(0, None)
axes[3].set_ylabel("$B(x)$ (G)")


# plot the current density
axes[4].plot(
    x,
    gle.current_density(x, *args) / 1e11,
    "-",
    zorder=1,
)
axes[4].plot(
    x,
    J(x, args[0], lambda_eff.max()) / 1e11,
    ":",
    color="C1",
    zorder=0,
)
axes[4].plot(
    x,
    J(x, args[0], lambda_eff.min()) / 1e11,
    ":",
    color="C2",
    zorder=0,
)
axes[4].set_ylim(0, None)
axes[4].set_ylabel("$J(x)$ ($10^{11}$ A m$^{-2}$)")

# label/limit the x-axis
axes[-1].set_xlabel(r"$x$ (nm)")
axes[-1].set_xlim(0, 150)


fig.savefig("effect-on-sc-properties.pdf")

plt.show()
