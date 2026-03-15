"""sinque.py – sinc function: sin(x)/x, safe at x=0."""

import numpy as np


def sinque(x):
    """
    Evaluate sin(x)/x element-wise, returning 1 at x=0.

    Parameters
    ----------
    x : array-like
        Input argument(s).

    Returns
    -------
    s : np.ndarray
        sin(x)/x, with s[i] = 1 where |x[i]| <= eps.
    """
    x = np.asarray(x, dtype=float)
    s = np.ones(x.shape)
    ii = np.abs(x) > np.finfo(float).eps
    s[ii] = np.sin(x[ii]) / x[ii]
    return s
