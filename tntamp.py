"""tntamp.py – amplitude spectrum for an impulsive seismic source.

Direct port of Margrave toolbox ``tntamp.m``.
"""

import numpy as np


def tntamp(fnot: float, f: np.ndarray, m: int = 2) -> np.ndarray:
    """
    Return an amplitude spectrum appropriate for an impulsive source.

    The spectral shape is::

        ampspec = (1 - gaus) / (1 + (f / fnot)^m)

    where ``gaus = exp(-(f / fnot)^2)``.  This gives a spectrum that rises
    from zero at DC, peaks near *fnot*, and rolls off at high frequencies
    with a rate controlled by *m*.

    Parameters
    ----------
    fnot : float
        Dominant (peak) frequency in Hz.
    f : array-like
        Frequency sample vector (e.g. as returned by :func:`fftrl`).
    m : int, optional
        Exponent controlling the high-frequency roll-off steepness.
        Larger *m* → sharper roll-off.  Default 2.

    Returns
    -------
    ampspec : np.ndarray
        Amplitude spectrum, same shape as *f*.

    Examples
    --------
    >>> f = np.linspace(0, 125, 513)
    >>> a = tntamp(30.0, f)
    >>> f[np.argmax(a)]          # peak near 30 Hz
    """
    f = np.asarray(f, dtype=float)
    gaus = np.exp(-(f / fnot) ** 2)
    # Guard against division by zero at f=0 when fnot=0 (not a real use-case,
    # but defensive).  The numerator is also 0 there, so 0/1 = 0 is correct.
    denom = 1.0 + (f / fnot) ** m
    ampspec = (1.0 - gaus) / denom
    return ampspec
