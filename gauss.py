"""
gauss.py — Gaussian distribution sampled at given frequencies.

Translated from MATLAB gauss.m (CREWES / Margrave).
"""

import numpy as np


def gauss(f: np.ndarray, fnot: float, fwidth: float) -> np.ndarray:
    """
    Return a Gaussian distribution sampled at the frequencies in *f*.

    Parameters
    ----------
    f      : 1-D array of frequency samples (Hz)
    fnot   : centre frequency of the Gaussian (Hz)
    fwidth : 1/e half-width of the Gaussian (Hz)

    Returns
    -------
    g : ndarray, same shape as *f*
        g = exp( -((f - fnot) / fwidth)^2 )
    """
    f = np.asarray(f, dtype=float)
    return np.exp(-((f - fnot) / fwidth) ** 2)
