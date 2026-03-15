"""xcoord.py – build a uniform coordinate vector.

Direct port of Margrave toolbox ``xcoord.m``.
"""

import numpy as np


def xcoord(xstart: float, delx: float, nx) -> np.ndarray:
    """
    Compute a coordinate vector of length *nx* starting at *xstart*.

    Parameters
    ----------
    xstart : float
        First coordinate value.
    delx : float
        Coordinate increment.
    nx : int or array-like
        Number of points.  If an array is supplied, ``len(nx)`` is used
        (matching the MATLAB ``xcoord(xstart, delx, v)`` form).

    Returns
    -------
    x : np.ndarray, 1-D
        Coordinate vector ``[xstart, xstart+delx, ..., xstart+(n-1)*delx]``
        of length *n*.

    Examples
    --------
    >>> xcoord(0.0, 0.004, 5)
    array([0.   , 0.004, 0.008, 0.012, 0.016])
    >>> xcoord(0.0, 0.004, np.zeros(5))   # vector form
    array([0.   , 0.004, 0.008, 0.012, 0.016])
    """
    if np.ndim(nx) > 0:
        nx = len(nx)
    nx = int(nx)
    xmax = xstart + (nx - 1) * delx
    return np.linspace(xstart, xmax, nx)
