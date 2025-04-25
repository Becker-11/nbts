"""Crank-Nicolson example.

Adapted from: https://georg.io/2013/12/03/Crank_Nicolson

see also: https://math.stackexchange.com/a/3311598
"""

from datetime import datetime
import numpy as np
from scipy import sparse
from scipy.sparse import diags
import matplotlib.pyplot as plt
from model import D, k, c, q

class CNSolver:
    """Crank-Nicolson Solver for 1D Diffusion Problems.

    This class encapsulates the Crank-Nicolson method to solve diffusion equations
    in 1D with specified initial and boundary conditions.

    Attributes:
        D_u (float): Diffusion coefficient (in nm^2/s).
        u_0 (float): Initial concentration close to x = 0 (e.g., at. % nm).
        v_0 (float): Background concentration (e.g., at. % nm).
        t_h (float): Maximum time in hours.
        N_x (int): Number of spatial grid points.
        x_max (float): Maximum spatial boundary (nm).
        N_t (int): Number of time grid points.
        x_grid (np.ndarray): Spatial grid.
        t_grid (np.ndarray): Temporal grid.
        sigma_u (float): Proportionality term for Crank-Nicolson.
    """

    def __init__(self, T, u_0=1e3, v_0=1e1, t_h=48, x_max=500.0, N_x=1001, N_t=4001):
        self.T = T
        self.u_0 = u_0
        self.v_0 = v_0
        self.t_h = t_h
        self.x_max = x_max
        self.N_x = N_x
        self.N_t = N_t

        # Constants
        self.D_u = D(T)  # Diffusion coefficient (in nm^2/s)
        self.c_0 = v_0 + u_0  # Initial concentration

        # Spatial and temporal grids
        self.x_grid = np.linspace(0.0, x_max, N_x, dtype=np.double)
        self.dx = np.diff(self.x_grid)[0]

        self.s_per_h = 60.0 * 60.0
        self.t_max = t_h * self.s_per_h
        self.t_grid = np.linspace(0.0, self.t_max, N_t, dtype=np.double)
        self.dt = np.diff(self.t_grid)[0]

        # Stability parameter
        self.r = (self.D_u * self.dt) / (self.dx * self.dx)
        self.stability = "STABLE" if self.r <= 0.5 else "POTENTIAL OSCILLATIONS"

        # Crank-Nicolson proportionality term
        self.sigma = 0.5 * self.r


    def gen_sparse_matrices(self):      
        """Generate the sparse matrices "A" and "B" used by the Crank-Nicolson method.
        
        Args:
            N_x: Dimension of (square) matrices.
            sigma: The "nudging" parameter.
        
        Returns:
            The (sparse) matrices A and B.
        """
        
        # common sparse matrix parameters
        _offsets = [1, 0, -1]
        _shape = (self.N_x, self.N_x)
        _format = "csr"
        
        # define matrix A's elements
        _A_upper = [-self.sigma]
        _A_diag = [1 + self.sigma] + [1 + 2 * self.sigma] * (self.N_x - 2) + [1 + self.sigma]
        _A_lower = [-self.sigma]
        _A_elements = [_A_upper, _A_diag, _A_lower]
        
        # create matrix A
        _A = sparse.diags_array(
            _A_elements,
            offsets=_offsets,
            shape=_shape,
            format=_format,
        )
        
        # define matrix B's elements
        _B_upper = [self.sigma]
        _B_diag = [1 - self.sigma] + [1 - 2 * self.sigma] * (self.N_x - 2) + [1 - self.sigma]
        _B_lower = [self.sigma]
        _B_elements = [_B_upper, _B_diag, _B_lower]
        
        # create matrix A
        _B = sparse.diags_array(
            _B_elements,
            offsets=_offsets,
            shape=(self.N_x, self.N_x),
            format=_format,
        )
        
        # return both matrix A and B
        return _A, _B

    def get_oxygen_profile(self):
        """Solve the diffusion equation using the Crank-Nicolson method.

        Returns:
            np.ndarray: The solution record (time x space).
        """
        # Initial condition: Concentration is all in the first spatial bin
        U_initial = sparse.csr_array([self.v_0 / self.dx] + [0] * (self.N_x - 1))
        U_record = np.zeros((self.N_t, self.N_x), dtype=np.double)

        for i, t in enumerate(self.t_grid):
            if i == 0:
                # Record the initial condition
                U_record[i] = U_initial.toarray()
            else:
                # Source term (plane source at x = 0)
                f_vec = sparse.csr_array([q(t, self.T) * (self.dt / self.dx)] + [0] * (self.N_x - 1))

                # Generate matrices (could be precomputed if D_u is constant)
                A, B = self.gen_sparse_matrices()

                # Solve for the next time step
                U = sparse.csr_array(U_record[i - 1])
                U_new = sparse.linalg.spsolve(A, B @ U + f_vec)
                U_record[i] = U_new

        return U_record
    





# Example usage
if __name__ == "__main__":
    T = 273.15 + 120
    u_0 = 1000.0
    v_0 = 10.0
    t_h = 12

    s_per_min = 60.0
    min_per_h = 60.0
    s_per_h = s_per_min * min_per_h

    DTYPE = np.double
    D_u = D(T)  # diffusion coefficient (in nm^2 / s)
    c_0 = v_0 + u_0  # initial concentration close to x = 0 (e.g., at. % nm)

    # specify the grid
    x_max = 100.0  # boundary lengths: [0.0, x_max] (nm)
    N_x = 2001  # number of equally spaced grid points within spatial boundaries
    x_grid = np.linspace(0.0, x_max, N_x, dtype=DTYPE)
    dx = np.diff(x_grid)[0]

    t_max = t_h * s_per_h  # time domain lengths: [0.0, t_max] (s)
    N_t = 8001  # number of equally spaced grid points within temporal boundaries
    t_grid = np.linspace(0.0, t_max, N_t, dtype=DTYPE)
    dt = np.diff(t_grid)[0]

    cc = np.array([c(xx, t_max, T, u_0, v_0, 0.01) for xx in x_grid])

    solver = CNSolver(T, u_0, v_0, t_h, x_max, N_x, N_t)
    U_record = solver.get_oxygen_profile()
    print(solver.stability)


    figc, axc = plt.subplots(
        2,
        1,
        figsize=(4.8, 4.8),
        sharex=True,
        sharey=False,
        gridspec_kw=dict(height_ratios=[2, 1]),
        constrained_layout=True,
    )

    
    axc[0].plot(
        x_grid,
        cc,
        "-",
        label="Exact solution",
        zorder=1,
    )

    axc[0].plot(
        x_grid,
        U_record[-1],
        "--",
        label="Approximate solution\n(Crank-Nicolson)",
        zorder=1,
    )

    axc[1].plot(
        x_grid,
        U_record[-1] - cc,
        ":",
        zorder=1,
    )

    axc[0].legend()
    axc[-1].set_xlabel("$x$ (nm)")
    axc[0].set_ylabel("$c(x, t)$ (at. %)")
    axc[1].set_ylabel(r"$\mathrm{Approximate} - \mathrm{Exact}$")

    plt.show()



# def gen_sparse_matrices(
#     N_x: int,
#     sigma: float,
# ):
#     """Generate the sparse matrices "A" and "B" used by the Crank-Nicolson method.
    
#     Args:
#         N_x: Dimension of (square) matrices.
#         sigma: The "nudging" parameter.
    
#     Returns:
#         The (sparse) matrices A and B.
#     """
    
#     # common sparse matrix parameters
#     _offsets = [1, 0, -1]
#     _shape = (N_x, N_x)
#     _format = "csr"
    
#     # define matrix A's elements
#     _A_upper = [-sigma]
#     _A_diag = [1 + sigma] + [1 + 2 * sigma] * (N_x - 2) + [1 + sigma]
#     _A_lower = [-sigma]
#     _A_elements = [_A_upper, _A_diag, _A_lower]
    
#     # create matrix A
#     _A = sparse.diags_array(
#         _A_elements,
#         offsets=_offsets,
#         shape=_shape,
#         format=_format,
#     )
    
#     # define matrix B's elements
#     _B_upper = [sigma]
#     _B_diag = [1 - sigma] + [1 - 2 * sigma] * (N_x - 2) + [1 - sigma]
#     _B_lower = [sigma]
#     _B_elements = [_B_upper, _B_diag, _B_lower]
    
#     # create matrix A
#     _B = sparse.diags_array(
#         _B_elements,
#         offsets=_offsets,
#         shape=(N_x, N_x),
#         format=_format,
#     )
    
#     # return both matrix A and B
#     return _A, _B


# T = 273.15 + 120
# u_0 = 1e3
# v_0 = 1e1
# t_h = 48

# s_per_min = 60.0
# min_per_h = 60.0
# s_per_h = s_per_min * min_per_h


# DTYPE = np.double

# D_u = D(T)  # diffusion coefficent (in nm^2 / s)
# c_0 = v_0 + u_0  # initial concentration close to x = 0 (e.g., at. % nm)


# # specify the grid
# x_max = 500.0  # boundary lengths: [0.0, x_max] (nm)
# N_x = 10001  # number of equally spaced grid points within spatial boundaries
# x_grid = np.linspace(0.0, x_max, N_x, dtype=DTYPE)
# dx = np.diff(x_grid)[0]


# t_max = t_h * s_per_h  # time domain lengths: [0.0, t_max] (s)
# N_t = 4001  # number of equally spaced grid points within temporal boundaries
# t_grid = np.linspace(0.0, t_max, N_t, dtype=DTYPE)
# dt = np.diff(t_grid)[0]


# U_record = CNSolver.get_oxygen_profile(T)
# # time the execution of the loop
# # note: by using "sparse" vectors/matrices, the loop execution speed should be
# #       > 10 faster than when using "dense" representations!
# # end_time = datetime.now()
# # print(f"Loop duration: {end_time - start_time}")

# #
# figc, axc = plt.subplots(
#     2,
#     1,
#     figsize=(4.8, 4.8),
#     sharex=True,
#     sharey=False,
#     gridspec_kw=dict(height_ratios=[2, 1]),
#     constrained_layout=True,
# )


# cc = np.array([c(xx, t_max, T, u_0, v_0, 0.0) for xx in x_grid])

# axc[0].plot(
#     x_grid,
#     # cc / cc[0],
#     cc,
#     "-",
#     label="Exact solution",
#     zorder=1,
# )

# axc[0].plot(
#     x_grid,
#     # U_record[-1] / U_record[-1][0],
#     U_record[-1],
#     "--",
#     label="Approximate solution\n(Crank-Nicolson)",
#     zorder=1,
# )

# axc[1].plot(
#     x_grid,
#     U_record[-1] - cc,
#     ":",
#     # label="ratio (Crank-Nicolson / exact)",
#     zorder=1,
# )


# axc[0].legend()
# # axc[1].legend()

# # axc[0].set_xlim(0, None)
# # axc[0].set_ylim(0, None)

# axc[-1].set_xlabel("$x$ (nm)")
# axc[0].set_ylabel("$c(x, t)$ (at. %)")
# axc[1].set_ylabel(r"$\mathrm{Approximate} - \mathrm{Exact}$")


# plt.show()



# # Von Neumann stability analysis
# # stability requirement for the forward time-centered space (FTCS) method
# r = (D_u * dt) / (dx * dx)
# stability = "STABLE" if r <= 0.5 else "POTENTIAL OSCILLATIONS"

# # calculate the proportionality term used to evolve the concentration
# sigma = DTYPE(0.5 * r)


# # print some useful info to the terminal
# # print(f"r = {r:.3f} [{stability}]")
# # print(f"sigma_u = {sigma_u:.3f}")
# # print(f"dx = {dx:.3f}")
# # print(f"dt = {dt:.3f}")



# def get_oxygen_profile():
#     # specify the initial condition
#     # this setup correspond to everything concentrated in the first spatial bin/element
#     # note the division to get units of concentration (assuming c_0 is from a plane source)
#     U_initial = sparse.csr_array(
#         [v_0 / dx] + [0] * (N_x - 1),
#     )

#     # hold a record of each of the (spatial) solutions at each time step
#     U_record = np.zeros(shape=(N_t, N_x), dtype=DTYPE)

#     # time the execution of the loop
#     #start_time = datetime.now()

#     # loop over all subsequent time steps and solve the system iteratively
#     for i, t in enumerate(t_grid):

#         # special case for the very first
#         if i == 0:
#             # add the initial state to the "record" of solutions
#             # note: must convert "sparse" array to a "dense" one
#             U_record[i] = U_initial.toarray()

#         # solve the the next time step using the spatial "state" at the previous timestep
#         else:
#             # generate the source term contribution (i.e., f_vec's first element)
#             # (i.e., the source term is treated as a plane source)
#             f_vec = sparse.csr_array(
#                 [q(t, T) * (dt / dx)] + [0] * (N_x - 1),
#             )

#             # create the tridiagonal matrices at each time-step
#             # note: this is really only needed if the diffusivity is time-dependent;
#             #       otherwise, they could be initialized once outside of loop 
#             A_u, B_u = gen_sparse_matrices(N_x, sigma_u)
            
#             # use the previous "solution" to get to current one
#             # note: must convert the "dense" array in the record to a "sparse" one
#             #       for fast computation!
#             U = sparse.csr_array(U_record[i - 1])

#             # find the new (spatial) solution vector by solving the system of linear equations
#             # note: this returns a "dense" array
#             U_new = sparse.linalg.spsolve(A_u, B_u @ U + f_vec)

#             # append the new solution to the record
#             U_record[i] = U_new
    
#     return U_record



