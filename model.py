"""Oxygen diffusion in Nb.

This module contains functions implementing the oxygen dissolution and diffusion
model in Nb metal, as described in:
G. Ciovati, Appl. Phys. Lett. 89, 022507 (2006).
https://doi.org/10.1063/1.2220059
"""

import numpy as np
from scipy import constants, integrate


def D(
    T: float,
    D_0: float = 0.0138,
    E_A: float = 111530,
) -> float:
    r"""Diffusion coefficient for oxygen in Nb.

    Diffusion coefficient :math:`D` for (interstitial) oxygen in Nb,
    assumed to follow an Arrhenius temperature dependence.

    Args:
        T: Temperature (K).
        D_0: Pre-exponential factor (cm\ :sup:`2` s\ :sup:`-1`\ ).
        E_A: Activation energy (J mol\ :sup:`-1` K\ :sup:`-1`\ ).

    Returns:
        The diffusion coefficient (nm\ :sup:`2` s\ :sup:`-1`\ ).
    """

    # unit conversion factor
    nm_per_cm = 1e7
    nm2_per_cm2 = nm_per_cm**2

    return D_0 * np.exp(-E_A / (constants.R * T)) * nm2_per_cm2


def k(
    T: float,
    A: float = 3.0e9,
    E_A: float = 135000,
) -> float:
    r"""Rate constant for oxygen dissolution from Nb's surface oxide.

    Rate constant :math:`k` for the dissolution of Nb's native surface oxide
    layer, assumed to follow an Arrhenius temperature dependence.

    Args:
        T: Temperature (K).
        A: Pre-exponential factor (s\ :sup:`-1`\ ).
        E_A: Activation energy (J mol\ :sup:`-1` K\ :sup:`-1`\ ).

    Returns:
        The rate constant (s\ :sup:`-1`\ ).
    """

    # evaluate the rate constant
    return A * np.exp(-E_A / (constants.R * T))


def q(
    t: float,
    T: float,
    u_0: float = 1e3,
) -> float:
    r"""Reaction term in the diffusion equation.

    Reaction term in the diffusion equation :math:`q(x, t, T)` for the rate
    that interstitial oxygen is incorporated into the system
    [see Eqs. (5) and (6) in Ciovati (2006)].

    Args:
        t: Time (s).
        T: Temperature (K).
        u_0: Total amount of surface oxygen (at. % nm).

    Returns:
        The oxygen introduction rate (at. % / s).
    """

    # evaluate the rate of oxygen incorporation into the system
    return u_0 * k(T) * np.exp(-k(T) * t)


def v(
    x: float,
    t: float,
    T: float,
    v_0: float = 1e1,
    c_0: float = 0.0,
) -> float:
    r"""Solution to the diffusion equation for interstitial oxygen.

    Solution to the diffusion equation :math:`v(x,t)` for interstitial oxygen
    initially present at the metal/surface oxide boundary
    [see Eq. (8) in Ciovati (2006)].

    Args:
        x: Depth (nm).
        t: Time (s).
        T: Temperature (K).
        v_0: Total amount of interstitial oxygen initially localized at the oxide interface (at. % nm).
        c_0: Concentration of interstitial oxygen uniformly distributed throughout the bulk (at. %).

    Returns:
        The oxygen concentration (at. %).
    """

    # argument for the exponential in the Gaussian
    arg = np.square(x) / (4.0 * D(T) * t)

    # prefactor for the Gaussian
    pre = v_0 / np.sqrt(np.pi * D(T) * t)

    # evaluate the Gaussian
    return pre * np.exp(-arg) + c_0


def _u_integrand(
    s: float,
    x: float,
    t: float,
    T: float,
) -> float:
    r"""Integrand from Eq. (7) in Ciovati (2006).

    Args:
        s: Integration variable (s).
        x: Depth (nm).
        t: Time (s).
        T: Temperature (K).

    Returns:
        The integrand.
    """

    # argument for the exponential
    arg = np.square(x) / (4.0 * D(T) * (t - s))

    # prefactor for the solution
    pre = k(T) * np.exp(-k(T) * s) / np.sqrt(t - s)

    return pre * np.exp(-arg)


def u(
    x: float,
    t: float,
    T: float,
    u_0: float = 1e3,
) -> float:
    r"""Solution to the diffusion equation for dissolved oxygen.

    Solution to the diffusion equation :math:`u(x,t)` for oxygen that is
    thermally dissolved from Nb's native surface oxide layer
    [see Eq. (7) in Ciovati (2006)].

    Args:
        x: Depth (nm).
        t: Time (s).
        T: Temperature (K).
        u_0: Total amount of surface oxygen (at. % nm).

    Returns:
        The oxygen concentration (at. %).
    """

    # evaluate the integral
    integral, _ = integrate.quad(
        _u_integrand,
        0,  # lower integration limit
        t,  # upper integration limit
        args=(x, t, T),
        full_output=False,
        epsabs=np.sqrt(np.finfo(float).eps),  # default = 1.49e-8
        epsrel=np.sqrt(np.finfo(float).eps),  # default = 1.49e-8
        limit=int(1e3),  # max sub-intervals used by adaptive algorithm (default = 50)
    )

    # prefactor for the solution
    pre = u_0 / np.sqrt(np.pi * D(T))

    # evaluate the Gaussian
    return pre * integral


def c(
    x: float,
    t: float,
    T: float,
    u_0: float = 1e3,
    v_0: float = 1e1,
    c_0: float = 0.0,
) -> float:
    """Total interstitial oxygen concentration in Nb metal.

    Total oxygen concentration, given by the sum of contributions from oxygen
    thermally dissolved from Nb's surface oxide layer and interstitial oxygen
    initially present at the metal/oxide interface
    [see Eq. (10) in Ciovati (2006)].

    Args:
        x: Depth (nm).
        t: Time (s).
        T: Temperature (K).
        u_0: Total amount of surface oxygen (at. % nm).
        v_0: Total amount of interstitial oxygen initially localized at the oxide interface (at. % nm).
        c_0: Concentration of interstitial oxygen uniformly distributed throughout the bulk (at. %).

    Returns:
        The total oxygen concentration (at. %).
    """

    # sum the u & v terms to get the total concentration
    return u(x, t, T, u_0) + v(x, t, T, v_0, c_0)
