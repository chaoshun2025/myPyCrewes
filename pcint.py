"""pcint.py – piecewise-constant (staircase) interpolation.

Mirrors the MATLAB ``pcint`` from the Margrave CREWES toolbox.
"""

import numpy as np


def pcint(x, y, xi):
    """
    Piecewise-constant (staircase) interpolation.

    Assumes that ``(x, y)`` represent a blocky / staircase function: for
    any query point ``xi`` between ``x[k]`` and ``x[k+1]`` the function
    evaluates to ``y[k]`` (left-step convention).  Points outside the
    range of ``x`` are extrapolated with the nearest endpoint value:

    * ``xi < x[0]``   →  ``y[0]``
    * ``xi > x[-1]``  →  ``y[-1]``

    Parameters
    ----------
    x : array_like
        Knot x-coordinates.  Should be sorted in ascending order (the MATLAB
        original documents this requirement); this implementation also handles
        unsorted ``xi`` correctly via ``numpy.searchsorted``.
    y : array_like
        Knot y-values, same length as ``x``.
    xi : array_like
        Query x-coordinates (any shape).

    Returns
    -------
    yi : np.ndarray
        Interpolated values, same shape as ``xi``.

    Notes
    -----
    The MATLAB implementation uses a sequential search loop over interior
    points.  This implementation uses :func:`numpy.searchsorted` for
    equivalent results in O(n log n) time regardless of the size of ``xi``.

    For an interior point ``xi`` that equals a knot exactly, ``y[k]`` is
    returned (matching MATLAB's ``find(x <= xi(k))`` → ``y(ii(end))``
    convention).

    Examples
    --------
    >>> import numpy as np
    >>> x  = np.array([1., 2., 3., 4.])
    >>> y  = np.array([10., 20., 30., 40.])
    >>> xi = np.array([0., 1., 1.5, 2., 3.5, 5.])
    >>> pcint(x, y, xi)
    array([10., 10., 10., 20., 30., 40.])
    """
    x  = np.asarray(x,  dtype=float).ravel()
    y  = np.asarray(y,  dtype=float).ravel()
    xi = np.asarray(xi, dtype=float)

    # searchsorted with side='right' gives the insertion index after any
    # existing equal values.  Subtracting 1 gives the index of the largest
    # x[k] that is <= xi (left-step convention).
    # Clipping to [0, len(y)-1] handles both extrapolation cases:
    #   xi < x[0]  → index -1 → clipped to 0 → y[0]
    #   xi > x[-1] → index len(x) - 1 → already at last → y[-1]
    idx = np.searchsorted(x, xi, side="right") - 1
    idx = np.clip(idx, 0, len(y) - 1)
    return y[idx]
