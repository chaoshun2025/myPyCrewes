"""ricker.py – generate a normalised Ricker (Mexican hat) wavelet."""

import numpy as np
from wavenorm import wavenorm


def ricker(dt, fdom=15.0, tlength=None):
    """
    Create a Ricker wavelet.

    The wavelet is centred at t=0 and normalised so that a sinusoid at the
    dominant frequency passes with unit amplitude (``wavenorm`` flag 2),
    exactly matching the MATLAB ``ricker.m`` behaviour.

    Parameters
    ----------
    dt : float
        Sample interval in seconds.
    fdom : float, optional
        Dominant frequency in Hz.  Default 15 Hz.
    tlength : float or None, optional
        Total wavelet duration in seconds.
        Default: 127 * dt  (so that length = 128 samples, a power of 2).

    Returns
    -------
    wavelet : np.ndarray, 1-D
        The Ricker wavelet (column vector convention internally).
    tw : np.ndarray, 1-D
        Time coordinate vector (symmetric around 0).

    Notes
    -----
    MATLAB convention for the zero-time sample placement:

    * n  = round(tlength / dt) + 1
    * nzero = ceil((n+1)/2)  → 1-based index of the t=0 sample
    * nl = nzero - 1         → samples to the left of t=0
    * nr = n - nzero         → samples to the right of t=0

    So for an even n the wavelet is slightly asymmetric (one extra sample on
    the left), matching MATLAB.
    """
    if tlength is None:
        tlength = 127.0 * dt

    n = int(round(tlength / dt)) + 1
    nzero = int(np.ceil((n + 1) / 2))  # 1-based → 0-based: nzero-1
    nr = n - nzero          # samples to the right
    nl = n - nr - 1         # samples to the left

    tw = dt * np.arange(-nl, nr + 1, dtype=float)

    pf = (np.pi * fdom) ** 2
    wavelet = (1.0 - 2.0 * pf * tw ** 2) * np.exp(-pf * tw ** 2)

    wavelet = wavenorm(wavelet, tw, 2)
    return wavelet, tw
