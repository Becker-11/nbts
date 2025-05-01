"""Chemical species involved in Nb's surface oxide dissolution.

This module contains the time-dependent concentrations of the "oxide" species
destroyed/created upon the dissolution of Nb's surface oxide layer.
"""

from typing import Annotated, Sequence
import numpy as np
from scipy import constants


def _k(
    T: Sequence[float],
    A: Annotated[float, 0:None],
    E_A: Annotated[float, 0:None],
) -> Sequence[float]:
    r"""Generic expression for an Arrhenius rate constant.

    Generic expression for the temperature dependence of a rate constant
    :math:`k`\ , assumed to follow an Arrhenius form.

    Args:
        T: Temperature (K).
        A: Pre-exponential factor (s\ :sup:`-1`\ ).
        E_A: Activation energy (kJ mol\ :sup:`-1` K\ :sup:`-1`\ ).

    Returns:
        The rate constant :math:`k_{1}` (s\ :sup:`-1`\ ).
    """

    # unit conversion
    J_per_kJ = 1e3

    # exponential arguments
    arg = (E_A * J_per_kJ) / (constants.R * T)

    # evaluate the rate constant
    return A * np.exp(-arg)


def k_1(
    T: Sequence[float],
    A: Annotated[float, 0:None] = 9.75e8,
    E_A: Annotated[float, 0:None] = 133.7,
) -> float:
    r"""Rate constant for the dissolution Nb2O5.

    Rate constant :math:`k_{1}` for the dissolution of Nb2O5 in Nb's native
    surface oxide layer, assumed to follow an Arrhenius temperature dependence.

    Args:
        T: Temperature (K).
        A: Pre-exponential factor (s\ :sup:`-1`\ ).
        E_A: Activation energy (kJ mol\ :sup:`-1` K\ :sup:`-1`\ ).

    Returns:
        The rate constant (s\ :sup:`-1`\ ).
    """

    # evaluate the rate constant
    return _k(T, A, E_A)


def k_2(
    T: Sequence[float],
    A: Annotated[float, 0:None] = 0.15e12,
    E_A: Annotated[float, 0:None] = 177.0,
) -> float:
    r"""Rate constant for the dissolution NbO2.

    Rate constant :math:`k_{2}` for the dissolution of NbO2 in Nb's native
    surface oxide layer, assumed to follow an Arrhenius temperature dependence.

    Args:
        T: Temperature (K).
        A: Pre-exponential factor (s\ :sup:`-1`\ ).
        E_A: Activation energy (kJ mol\ :sup:`-1` K\ :sup:`-1`\ ).

    Returns:
        The rate constant :math:`k_{2}` (s\ :sup:`-1`\ ).
    """

    # evaluate the rate constant
    return _k(T, A, E_A)


def D(
    T: Sequence[float],
    D_0: Annotated[float, 0:None] = 0.59e-6,
    E_A: Annotated[float, 0:None] = 109.7,
) -> Sequence[float]:
    r"""Diffusion coefficient for interstital oxygen in Nb.

    Diffusion coefficient :math:`D` for interstital oxygen impurities in Nb
    metal. Its temperature dependence is assumed to follow an Arrhenius form.

    Args:
        T: Temperature (K).
        D_0: Pre-exponential factor (m\ :sup:`2` s\ :sup:`-1`\ ).
        E_A: Activation energy (kJ mol\ :sup:`-1` K\ :sup:`-1`\ ).

    Returns:
        The rate constant :math:`k_{1}` (s\ :sup:`-1`\ ).
    """

    # evaluate the diffusiviy
    return _k(T, D_0, E_A)


def c_Nb2O5(
    t: Sequence[float],
    T: Annotated[float, 0:None],
    c_Nb2O5_0: Annotated[float, 0:None],
) -> Sequence[float]:
    """Concentration of Nb2O5.

    Args:
        t: Time (s).
        T: Temperature (K).
        c_Nb2O5_0: Initial concentration of Nb2O5 (at. %).

    Returns:
        The concentration of Nb2O5 (at. %).
    """

    # evaluate the rate constant
    rate_1 = k_1(T)

    # evaluate the concentration
    return c_Nb2O5_0 * np.exp(-rate_1 * t)


def _c_common(
    t: Sequence[float],
    T: Annotated[float, 0:None],
    c_Nb2O5_0: Annotated[float, 0:None],
    c_NbO2_0: Annotated[float, 0:None] = 0.0,
) -> Sequence[float]:
    """Common concentration term.

    Args:
        t: Time (s).
        T: Temperature (K).
        c_Nb2O5_0: Initial concentration of Nb2O5 (at. %).
        c_NbO2_0: Initial concentration of NbO2 (at. %).

    Returns:
        The common concentration term (at. %).
    """

    # evaluate the rate constants
    rate_1 = k_1(T)
    rate_2 = k_2(T)

    # evaluate the common term in the concentration expresssions
    return (c_NbO2_0 * rate_2 - (c_NbO2_0 + 2 * c_Nb2O5_0) * rate_1) * np.exp(
        -rate_2 * t
    )


def c_NbO2(
    t: Sequence[float],
    T: Annotated[float, 0:None],
    c_Nb2O5_0: Annotated[float, 0:None],
    c_NbO2_0: Annotated[float, 0:None] = 0.0,
) -> Sequence[float]:
    """Concentration of NbO2.

    Args:
        t: Time (s).
        T: Temperature (K).
        c_Nb2O5_0: Initial concentration of Nb2O5 (at. %).
        c_NbO2_0: Initial concentration of NbO2 (at. %).

    Returns:
        The concentration of NbO2 (at. %).
    """

    # evaluate the rate constants
    rate_1 = k_1(T)
    rate_2 = k_2(T)

    # break up the terms into the denominator & numerator
    denominator = rate_2 - rate_1
    numerator = _c_common(t, T, c_Nb2O5_0, c_NbO2_0) + 2 * c_Nb2O5_0 * rate_1 * np.exp(
        -rate_1 * t
    )

    # evaluate the concentration
    return numerator / denominator


def c_NbO(
    t: Sequence[float],
    T: Annotated[float, 0:None],
    c_Nb2O5_0: Annotated[float, 0:None],
    c_NbO2_0: Annotated[float, 0:None] = 0.0,
    c_NbO_0: Annotated[float, 0:None] = 0.0,
) -> Sequence[float]:
    """Concentration of NbO.

    Args:
        t: Time (s).
        T: Temperature (K).
        c_Nb2O5_0: Initial concentration of Nb2O5 (at. %).
        c_NbO2_0: Initial concentration of NbO2 (at. %).
        c_NbO_0: Initial concentration of NbO (at. %).

    Returns:
        The concentration of NbO (at. %).
    """

    # evaluate the rate constants
    rate_1 = k_1(T)
    rate_2 = k_2(T)

    # maximum total concentration
    c_NbO_max = 2 * c_Nb2O5_0 + c_NbO2_0 + c_NbO_0

    # evaluate the numerator/denominator separately
    denominator = rate_2 - rate_1
    numerator = _c_common(t, T, c_Nb2O5_0, c_NbO2_0) + 2 * c_Nb2O5_0 * rate_2 * np.exp(
        -rate_1 * t
    )

    # evaluate the concentration
    return c_NbO_max - (numerator / denominator)


def c_O(
    t: Sequence[float],
    T: Annotated[float, 0:None],
    c_Nb2O5_0: Annotated[float, 0:None],
    c_NbO2_0: Annotated[float, 0:None] = 0.0,
    c_O_0: Annotated[float, 0:None] = 0.0,
) -> Sequence[float]:
    """Concentration of O.

    Args:
        t: Time (s).
        T: Temperature (K).
        c_Nb2O5_0: Initial concentration of Nb2O5 (at. %).
        c_NbO2_0: Initial concentration of NbO2 (at. %).
        c_O_0: Initial concentration of O (at. %).

    Returns:
        The concentration of O (at. %).
    """

    # evaluate the rate constants
    rate_1 = k_1(T)
    rate_2 = k_2(T)

    # maximum total concentration
    c_O_max = 3 * c_Nb2O5_0 + c_NbO2_0 + c_O_0

    # evaluate the numerator/denominator separately
    denominator = rate_2 - rate_1
    numerator = _c_common(t, T, c_Nb2O5_0, c_NbO2_0) + c_Nb2O5_0 * (
        3 * rate_2 - rate_1
    ) * np.exp(-rate_1 * t)

    # evaluate the concentration
    return c_O_max - (numerator / denominator)


def dc_O_dt(
    t: Sequence[float],
    T: Annotated[float, 0:None],
    c_Nb2O5_0: Annotated[float, 0:None],
    c_NbO2_0: Annotated[float, 0:None] = 0.0,
) -> Sequence[float]:
    """Time derivative of the concentration of O.

    Args:
        t: Time (s).
        T: Temperature (K).
        c_Nb2O5_0: Initial concentration of Nb2O5 (at. %).
        c_NbO2_0: Initial concentration of NbO2 (at. %).

    Returns:
        The time derivative of the concentration of O (at. %).
    """

    # evaluate the rate constants
    rate_1 = k_1(T)
    rate_2 = k_2(T)

    # evaluate the numerator/denominator separately
    denominator = rate_2 - rate_1
    numerator = rate_2 * _c_common(t, T, c_Nb2O5_0, c_NbO2_0) + rate_1 * c_Nb2O5_0 * (
        3 * rate_2 - rate_1
    ) * np.exp(-rate_1 * t)

    # evaluate the first time-derivative of the concentration
    return numerator / denominator
