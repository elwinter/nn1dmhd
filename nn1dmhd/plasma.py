"""Routines for common plasma computations.

This module provides functions which perform common calculations for
plasmas.

The default mode of computation uses the commonly-accepted numerical
values for physical constants. If the "normalized" flag is set in any
function, all calculations are performed with physical constants set
to unity.

Author
------
Eric Winter (eric.winter62@gmail.com)

"""


import numpy as np
import scipy.constants as spc


def electron_plasma_angular_frequency(n:float, normalize:bool=False):
    """Compute the electron plasma angular frequency.

    Compute the electron plasma angular frequency.

    Parameters
    ----------
    n : np.ndarray
        Electron number density (units of m**-3).
    normalize : bool (default False)
        Assume unit values for all physical constants.

    Returns
    -------
    wp : float
        Electron plasma angular frequency (units of rad/s).
    """
    if normalize:
        wp = np.sqrt(n)
    else:
        wp = np.sqrt(n*spc.e**2/(spc.epsilon_0*spc.m_e))
    return wp


def electron_thermal_speed(T:float, normalize:bool=False):
    """Compute the electron thermal speed.

    Compute the electron thermal speed. This is defined as the RMS
    speed along any single direection.

    Parameters
    ----------
    T : np.ndarray
        Temperature (units of K).
    normalize : bool (default False)
        Assume unit values for all physical constants.

    Returns
    -------
    vth : float
        Electron thermal speed (units of m/s).

    """
    if normalize:
        vth = np.sqrt(2*T)
    else:
        vth = np.sqrt(2*spc.k*T/spc.m_e)
    return vth


def electron_plasma_wave_angular_frequency(n:float, T:float, k:float, normalize:bool=False):
    """Compute the electron plasma wave angular frequency.

    Compute the electron plasma wave angular frequency.

    Parameters
    ----------
    n : np.ndarray
        Electron number density (units of m**-3).
    T : float
        Temperature (units of K).
    k : np.ndarray
        Wavenumber (units of rad/m).
    normalize : bool (default False)
        Assume unit values for all physical constants.

    Returns
    -------
    w : float
        Electron plasma wave angular frequency (units of rad/s).
    """
    wp = electron_plasma_angular_frequency(n, normalize=normalize)
    vth = electron_thermal_speed(T, normalize=normalize)
    w = np.sqrt(wp**2 + 1.5*vth**2*k**2)
    return w


def electron_plasma_wave_phase_speed(n:float, T:float, k:float, normalize:bool=False):
    """Compute the electron plasma wave phase speed.

    Compute the electron plasma wave phase speed.

    Parameters
    ----------
    n : np.ndarray
        Electron number density (units of m**-3).
    T : float
        Temperature (units of K).
    k : np.ndarray
        Wavenumber (units of rad/m).
    normalize : bool (default False)
        Assume unit values for all physical constants.

    Returns
    -------
    vphase : float
        Phase speed (units of m/s).
    """
    w = electron_plasma_wave_angular_frequency(n, T, k, normalize=normalize)
    vphase = w/k
    return vphase
