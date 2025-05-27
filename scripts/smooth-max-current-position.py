import numpy as np
import pandas as pd
from scipy import interpolate, optimize
import matplotlib.pyplot as plt


# read the table of simulated values
data = pd.read_csv("baking-effect-on-sc-properties.csv")

# naively extract argmax{J(x)} from the table
x_max = data["Depth (nm)"].to_numpy()[np.argmax(data["Current Density (A/m^2)"])]

# create a smooth interpolation of J(x) data
j_smooth = interpolate.Akima1DInterpolator(
    data["Depth (nm)"],
    data["Current Density (A/m^2)"],
    method="akima",
    extrapolate=False,
)

# create a smooth interpolation of dJ(x)/dx
deriv_j_smooth = j_smooth.derivative(1)

# find the roots of dJ(x)/dx [if any] - `roots` is a numpy array (empty if none found)
roots = deriv_j_smooth.roots(extrapolate=False)


# figure/axes for plotting J(x) and its derivative
fig, axes = plt.subplots(
    2,
    1,
    sharex=True,
    sharey=False,
    figsize=(4.8, 4.8),
    constrained_layout=True,
)

# "dummy" points for plotting
x = np.linspace(
    0.0,
    min(100.0, data["Depth (nm)"].max()),
    500,
)

# plot the "smoothed" J(x)
axes[0].plot(x, j_smooth(x), "-", zorder=1)

# plot the "smoothed" dJ(x)/dx
axes[1].plot(x, deriv_j_smooth(x), "-", zorder=1)

# mark the "zero" crossing in the dJ(x)/dx plot
axes[1].axhline(0.0, linestyle=":", color="lightgrey", zorder=0)

# add annotated vertical lines to each axis denoting the minima/maxima in J(x) [if any]
for a in axes:
    for r in roots:
        a.axvline(
            r,
            linestyle=":",
            color="lightgrey",
            zorder=0,
        )
        if a == axes[0]:
            a.text(
                r,
                min(axes[0].get_ylim()) + 0.25 * np.diff(axes[0].get_ylim()),
                f"$x = {r:.3f}$ nm",
                ha="center",
                va="center",
                size="x-small",
                color="C0",
                bbox=dict(boxstyle="round", edgecolor="white", facecolor="white"),
                rotation=90,
                zorder=1,
            )

# label the naive argmax{J(x)} value
axes[0].text(
    x_max,
    min(axes[0].get_ylim()) + 1.15 * np.diff(axes[0].get_ylim()),
    f"naive maximum $x = {x_max:.3f}$ nm",
    ha="center",
    va="center",
    size="x-small",
    color="C0",
)

# don't show negative depths
axes[-1].set_xlim(0, None)

# label the axes
axes[-1].set_xlabel(r"$x$ (nm)")
axes[-1].set_ylabel(r"$\mathrm{d} J(x) \, / \, \mathrm{d}x$")
axes[0].set_ylabel(r"$J(x)$ (A m$^{-2}$)")

plt.show()
