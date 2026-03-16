"""interpextrap.py – linear interpolation with linear end-extrapolation.

Mirrors the MATLAB ``interpextrap`` from the Margrave CREWES toolbox.
"""

import numpy as np
from lint import lint


def interpextrap(x, y, x0, flag=1):
    """
    Linear interpolation with optional linear or constant extrapolation.

    Identical to ``np.interp`` / ``lint`` for query points inside
    ``[min(x), max(x)]``.  For points outside the bounds, extrapolation is
    performed using the slope of the nearest end segment (``flag=1``) or a
    constant (horizontal) extension (``flag=0``).

    Parameters
    ----------
    x : array_like
        Knot x-coordinates.  May be in ascending or descending order.
    y : array_like
        Knot y-values, same length as ``x``.
    x0 : array_like
        Query x-coordinates (any shape).
    flag : int, optional
        Extrapolation mode:

        * ``1`` – linear extrapolation using the slope of the nearest
          end segment.  *(default)*
        * ``0`` – constant (horizontal) extrapolation; the end value is
          held flat.

    Returns
    -------
    y0 : np.ndarray
        Interpolated / extrapolated values, same shape as ``x0``.

    Notes
    -----
    The function handles both ascending (``x[0] < x[-1]``) and descending
    (``x[0] > x[-1]``) knot sequences, matching the MATLAB original which
    has two branches for these cases.
    """
    x  = np.asarray(x,  dtype=float).ravel()
    y  = np.asarray(y,  dtype=float).ravel()
    x0 = np.asarray(x0, dtype=float)

    nx  = len(x)
    y0  = np.full(x0.shape, np.nan, dtype=float)

    if nx == 1:
        y0[:] = y[0]
        return y0

    # ------------------------------------------------------------------ #
    # Extrapolation beyond the endpoints
    # ------------------------------------------------------------------ #
    if x[0] < x[-1]:                   # ascending order
        # --- beginning (x0 < x[0]) ---
        ind = x0 < x[0]
        if np.any(ind):
            m1 = (y[1] - y[0]) / (x[1] - x[0]) if flag else 0.0
            y0[ind] = m1 * (x0[ind] - x[0]) + y[0]

        # --- end (x0 > x[-1]) ---
        ind = x0 > x[-1]
        if np.any(ind):
            m2 = (y[-1] - y[-2]) / (x[-1] - x[-2]) if flag else 0.0
            y0[ind] = m2 * (x0[ind] - x[-1]) + y[-1]

    else:                               # descending order
        # --- beginning (x0 > x[0]) ---
        ind = x0 > x[0]
        if np.any(ind):
            m1 = (y[1] - y[0]) / (x[1] - x[0]) if flag else 0.0
            y0[ind] = m1 * (x0[ind] - x[0]) + y[0]

        # --- end (x0 < x[-1]) ---
        ind = x0 < x[-1]
        if np.any(ind):
            m2 = (y[-1] - y[-2]) / (x[-1] - x[-2]) if flag else 0.0
            y0[ind] = m2 * (x0[ind] - x[-1]) + y[-1]

    # ------------------------------------------------------------------ #
    # Interior interpolation (fills remaining NaNs)
    # ------------------------------------------------------------------ #
    interior = np.isnan(y0)
    if np.any(interior):
        xi_interior = x0[interior]
        y0[interior] = lint(x, y, xi_interior)

    return y0
