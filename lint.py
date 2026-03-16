"""lint.py – linear interpolation (legacy alias for pwlint).

Identical to :func:`pwlint`.  Retained for historical compatibility with
the Margrave CREWES toolbox.  Prefer :func:`pwlint` for new code.
"""

import numpy as np
from between import between


def lint(x, y, xi):
    """
    Linear interpolation.

    Identical to :func:`pwlint`.  Use when ``len(x)`` is much smaller than
    ``len(xi)``; otherwise prefer ``np.interp``.

    Parameters
    ----------
    x : array_like
        Knot x-coordinates (monotonically increasing).
    y : array_like
        Knot y-values, same length as ``x``.
    xi : array_like
        Query x-coordinates.

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
    nx = len(x)

    for k in range(nx - 1):
        ii = between(x[k], x[k + 1], xi, flag=2)
        if len(ii) > 0:
            slope = (y[k + 1] - y[k]) / (x[k + 1] - x[k])
            yi[ii] = y[k] + slope * (xi[ii] - x[k])

    return yi
