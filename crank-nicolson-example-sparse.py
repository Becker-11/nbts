"""Crank-Nicolson example.

Adapted from: https://georg.io/2013/12/03/Crank_Nicolson

see also: https://math.stackexchange.com/a/3311598
"""

from datetime import datetime
import numpy as np
from scipy import sparse
from scipy.sparse import diags_array
import matplotlib.pyplot as plt
from model import D, k, c, q


def gen_sparse_matrices(
    N_x: int,
    sigma: float,
):
    """Generate the sparse matrices "A" and "B" used by the Crank-Nicolson method.
    
    Args:
        N_x: Dimension of (square) matrices.
        sigma: The "nudging" parameter.
    
    Returns:
        The (sparse) matrices A and B.
    """
    
    # common sparse matrix parameters
    _offsets = [1, 0, -1]
    _shape = (N_x, N_x)
    _format = "csr"
    
    # define matrix A's elements
    _A_upper = [-sigma]
    _A_diag = [1 + sigma] + [1 + 2 * sigma] * (N_x - 2) + [1 + sigma]
    _A_lower = [-sigma]
    _A_elements = [_A_upper, _A_diag, _A_lower]
    
    # create matrix A
    _A = sparse.diags_array(
        _A_elements,
        offsets=_offsets,
        shape=_shape,
        format=_format,
    )
    
    # define matrix B's elements
    _B_upper = [sigma]
    _B_diag = [1 - sigma] + [1 - 2 * sigma] * (N_x - 2) + [1 - sigma]
    _B_lower = [sigma]
    _B_elements = [_B_upper, _B_diag, _B_lower]
    
    # create matrix A
    _B = sparse.diags_array(
        _B_elements,
        offsets=_offsets,
        shape=(N_x, N_x),
        format=_format,
    )
    
    # return both matrix A and B
    return _A, _B


T = 273.15 + 120
u_0 = 1e3
v_0 = 1e1
t_h = 48

s_per_min = 60.0
min_per_h = 60.0
s_per_h = s_per_min * min_per_h


DTYPE = np.double

D_u = D(T)  # diffusion coefficent (in nm^2 / s)
c_0 = v_0 + u_0  # initial concentration close to x = 0 (e.g., at. % nm)


# specify the grid
x_max = 500.0  # boundary lengths: [0.0, x_max] (nm)
N_x = 1001  # number of equally spaced grid points within spatial boundaries
x_grid = np.linspace(0.0, x_max, N_x, dtype=DTYPE)
dx = np.diff(x_grid)[0]


t_max = t_h * s_per_h  # time domain lengths: [0.0, t_max] (s)
N_t = 4001  # number of equally spaced grid points within temporal boundaries
t_grid = np.linspace(0.0, t_max, N_t, dtype=DTYPE)
dt = np.diff(t_grid)[0]


# Von Neumann stability analysis
# stability requirement for the forward time-centered space (FTCS) method
r = (D_u * dt) / (dx * dx)
stability = "STABLE" if r <= 0.5 else "POTENTIAL OSCILLATIONS"

# calculate the proportionality term used to evolve the concentration
sigma_u = DTYPE(0.5 * r)


# print some useful info to the terminal
print(f"r = {r:.3f} [{stability}]")
print(f"sigma_u = {sigma_u:.3f}")
print(f"dx = {dx:.3f}")
print(f"dt = {dt:.3f}")


# specify the initial condition
# this setup correspond to everything concentrated in the first spatial bin/element
# note the division to get units of concentration (assuming c_0 is from a plane source)
U_initial = sparse.csr_array(
    [v_0 / dx] + [0] * (N_x - 1),
)

# hold a record of each of the (spatial) solutions at each time step
U_record = np.zeros(shape=(N_t, N_x), dtype=DTYPE)

# time the execution of the loop
start_time = datetime.now()

# loop over all subsequent time steps and solve the system iteratively
for i, t in enumerate(t_grid):

    # special case for the very first
    if i == 0:
        # add the initial state to the "record" of solutions
        # note: must convert "sparse" array to a "dense" one
        U_record[i] = U_initial.toarray()

    # solve the the next time step using the spatial "state" at the previous timestep
    else:
        # generate the source term contribution (i.e., f_vec's first element)
        # (i.e., the source term is treated as a plane source)
        f_vec = sparse.csr_array(
            [q(t, T) * (dt / dx)] + [0] * (N_x - 1),
        )

        # create the tridiagonal matrices at each time-step
        # note: this is really only needed if the diffusivity is time-dependent;
        #       otherwise, they could be initialized once outside of loop 
        A_u, B_u = gen_sparse_matrices(N_x, sigma_u)
        
        # use the previous "solution" to get to current one
        # note: must convert the "dense" array in the record to a "sparse" one
        #       for fast computation!
        U = sparse.csr_array(U_record[i - 1])

        # find the new (spatial) solution vector by solving the system of linear equations
        # note: this returns a "dense" array
        U_new = sparse.linalg.spsolve(A_u, B_u @ U + f_vec)

        # append the new solution to the record
        U_record[i] = U_new


# time the execution of the loop
# note: by using "sparse" vectors/matrices, the loop execution speed should be
#       > 10 faster than when using "dense" representations!
end_time = datetime.now()
print(f"Loop duration: {end_time - start_time}")

#
figc, axc = plt.subplots(
    2,
    1,
    figsize=(4.8, 4.8),
    sharex=True,
    sharey=False,
    gridspec_kw=dict(height_ratios=[2, 1]),
    constrained_layout=True,
)


cc = np.array([c(xx, t_max, T, u_0, v_0, 0.0) for xx in x_grid])

axc[0].plot(
    x_grid,
    # cc / cc[0],
    cc,
    "-",
    label="Exact solution",
    zorder=1,
)

axc[0].plot(
    x_grid,
    # U_record[-1] / U_record[-1][0],
    U_record[-1],
    "--",
    label="Approximate solution\n(Crank-Nicolson)",
    zorder=1,
)

axc[1].plot(
    x_grid,
    U_record[-1] - cc,
    ":",
    # label="ratio (Crank-Nicolson / exact)",
    zorder=1,
)


axc[0].legend()
# axc[1].legend()

# axc[0].set_xlim(0, None)
# axc[0].set_ylim(0, None)

axc[-1].set_xlabel("$x$ (nm)")
axc[0].set_ylabel("$c(x, t)$ (at. %)")
axc[1].set_ylabel(r"$\mathrm{Approximate} - \mathrm{Exact}$")


plt.show()
