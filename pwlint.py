"""pwlint.py – piecewise-linear interpolation.

Much faster than a general-purpose interpolator when the knot vector ``x``
is short compared to the query vector ``xi``.  Points outside the range of
``x`` return ``NaN``, matching the MATLAB original.

Mirrors the MATLAB ``pwlint`` from the Margrave CREWES toolbox.
"""

import numpy as np
from between import between


def pwlint(x, y, xi):
    """
    Piecewise-linear interpolation.

    Parameters
    ----------
    x : array_like
        Knot x-coordinates (monotonically increasing).
    y : array_like
        Knot y-values, same length as ``x``.
    xi : array_like
        Query x-coordinates (any shape).

    Returns
    -------
    yi : np.ndarray
        Interpolated values, same shape as ``xi``.
        Points outside ``[x[0], x[-1]]`` are ``NaN``.
    """
    x  = np.asarray(x,  dtype=float).ravel()
    y  = np.asarray(y,  dtype=float).ravel()
    xi = np.asarray(xi, dtype=float)

    yi = np.full(xi.shape, np.nan, dtype=float)
    nsegs = len(x) - 1

    for k in range(nsegs):
        ii = between(x[k], x[k + 1], xi, flag=2)
        if len(ii) > 0:
            slope = (y[k + 1] - y[k]) / (x[k + 1] - x[k])
            yi[ii] = y[k] + slope * (xi[ii] - x[k])

    return yi
