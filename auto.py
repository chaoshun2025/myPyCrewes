import numpy as np


def auto(v, n=None, flag=1.0):
    """
    AUTO: single-sided autocorrelation

    a = auto(v, n, flag)

    Computes n lags of the one-sided autocorrelation of vector v.
    The first lag, a[0], is the 'zeroth lag'.

    Parameters
    ----------
    v : array_like
        Input 1-D vector.
    n : int, optional
        Number of lags desired (cannot exceed len(v)).
        Default = len(v).
    flag : float, optional
        1.0  -> normalize so that the zero-lag equals 1.
        else -> no normalization.
        Default = 1.0.

    Returns
    -------
    a : np.ndarray
        One-sided autocorrelation as a 1-D array. a[0] is the zero lag.
    """
    v = np.asarray(v, dtype=float).ravel()
    if n is None:
        n = len(v)
    if n > len(v):
        raise ValueError("n cannot be larger than len(v)")

    a = np.zeros(n)
    u = v.copy()
    for k in range(n):
        a[k] = np.dot(u, v)
        v = np.concatenate(([0.0], v[:-1]))

    if flag == 1.0:
        mx = np.max(a)
        if mx != 0:
            a = a / mx
    return a
