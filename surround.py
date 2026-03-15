"""surround.py – find indices where a vector brackets a scalar test value.

Direct port of Margrave toolbox ``surround.m``.
"""

import numpy as np


def surround(x: np.ndarray, xtest: float) -> np.ndarray:
    """
    Return indices where vector *x* brackets the scalar *xtest*.

    For each consecutive pair ``x[i], x[i+1]`` the index *i* is returned
    when the pair straddles *xtest* in either direction::

        x[i] <= xtest < x[i+1]   (ascending transition)
            or
        x[i] >= xtest > x[i+1]   (descending transition)

    This matches the MATLAB definition exactly, including:

    * returning an empty array when *xtest* is outside the range of *x*
    * returning multiple indices when *x* is non-monotonic and brackets
      *xtest* more than once
    * returning **0-based** indices (MATLAB returns 1-based)

    Parameters
    ----------
    x : array-like, 1-D
        Input vector.  Need not be monotonic.
    xtest : float
        Scalar test value to locate within *x*.

    Returns
    -------
    ind : np.ndarray of int
        0-based indices *i* such that ``x[i:i+2]`` brackets *xtest*.
        Empty array if *xtest* lies outside the range of *x*.

    Examples
    --------
    >>> surround(np.arange(1, 11), -1)
    array([], dtype=int64)
    >>> surround(np.arange(1, 11), 3)
    array([2])
    >>> x = np.concatenate([np.arange(1, 11), np.arange(9, 0, -1)])
    >>> surround(x, 3)
    array([ 2, 16])
    """
    x = np.asarray(x, dtype=float).ravel()
    x1 = x[:-1]
    x2 = x[1:]
    ind = np.where(
        ((x1 <= xtest) & (x2 > xtest)) |
        ((x1 >= xtest) & (x2 < xtest))
    )[0]
    return ind
