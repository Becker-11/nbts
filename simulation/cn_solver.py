"""Crank-Nicolson example.

Adapted from: https://georg.io/2013/12/03/Crank_Nicolson

see also: https://math.stackexchange.com/a/3311598
"""

import numpy as np
from scipy import sparse
from simulation.ciovati_model import D, q
import simulation.dissolution_species

class CNSolver:
    # TODO: fix docementation
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
        #self.D_u = D(T)  # Diffusion coefficient (in nm^2/s)
        self.D_u = None

        
        self.c_0 = v_0 + u_0  # Initial concentration
        # TODO: fix initial concentration to use dissolution_species
        #self.c_0 = dissolution_species.c_Nb2O5(0, T, )  # Initial concentration (Nb2O5)

        # Spatial and temporal grids
        self.x_grid = np.linspace(0.0, x_max, N_x, dtype=np.double)
        self.dx = np.diff(self.x_grid)[0]

        self.s_per_h = 60.0 * 60.0
        self.t_max = t_h * self.s_per_h
        self.t_grid = np.linspace(0.0, self.t_max, N_t, dtype=np.double)
        self.dt = np.diff(self.t_grid)[0]

        # # Stability parameter
        # self.r = (self.D_u * self.dt) / (self.dx * self.dx)
        # self.stability = "STABLE" if self.r <= 0.5 else "POTENTIAL OSCILLATIONS"

        # # Crank-Nicolson proportionality term
        # self.sigma = 0.5 * self.r


        self.r = None
        self.stability = None
        self.sigma = None


    def gen_sparse_matrices(self, i):      
        """Generate the sparse matrices "A" and "B" used by the Crank-Nicolson method.
        
        Args:
            N_x: Dimension of (square) matrices.
            sigma: The "nudging" parameter.
        
        Returns:
            The (sparse) matrices A and B.
        """
        # Initialize the diffusion coefficient for time t
        self.D_u = D(self.T[i])  # Diffusion coefficient (in nm^2/s)
        # Update the stability parameter
        self.r = (self.D_u * self.dt) / (self.dx * self.dx)
        # TODO: check if r is stable for the maximum T[i] for a given recipe
        # ...   I think right now it is only showing the last T[i] which will be for a low temperature
        self.stability = "STABLE" if self.r <= 0.5 else "POTENTIAL OSCILLATIONS"
        # Crank-Nicolson proportionality term
        self.sigma = 0.5 * self.r

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
                f_vec = sparse.csr_array([q(t, self.T[i]) * (self.dt / self.dx)] + [0] * (self.N_x - 1))

                # Generate matrices (could be precomputed if D_u is constant)
                A, B = self.gen_sparse_matrices(i)

                # Solve for the next time step
                U = sparse.csr_array(U_record[i - 1])
                U_new = sparse.linalg.spsolve(A, B @ U + f_vec)
                U_record[i] = U_new

        return U_record
    