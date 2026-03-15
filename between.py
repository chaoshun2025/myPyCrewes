import numpy as np


def between(x1, x2, testpts, flag=0):
    """
    BETWEEN: logical test — find samples in a vector between given bounds.

    indices = between(x1, x2, testpts, flag)

    Returns the indices of elements in testpts that lie between x1 and x2.
    If no points are found, returns an empty array (length 0).

    Parameters
    ----------
    x1, x2 : scalar
        Boundary values (order does not matter).
    testpts : array_like
        Array of values to test.
    flag : int, optional
        0 -> neither endpoint included  (x1 < . < x2)
        1 -> x1 included                (x1 <= . < x2)
        2 -> both endpoints included    (x1 <= . <= x2)
        Default = 0.

    Returns
    -------
    indices : np.ndarray of int
        Zero-based indices of testpts elements that satisfy the condition.
        Returns an empty array (np.array([], dtype=int)) when nothing found.

    Notes
    -----
    The original MATLAB version returns a scalar 0 when nothing is found.
    Here we return an empty integer array for cleaner NumPy idioms; callers
    should test with `len(indices) == 0` rather than `indices == 0`.
    """
    if np.size(x1) != 1 or np.size(x2) != 1:
        raise ValueError("x1 and x2 must be scalars")

    testpts = np.asarray(testpts)
    lo, hi = (x1, x2) if x1 < x2 else (x2, x1)

    if flag == 0:
        mask = (testpts > lo) & (testpts < hi)
    elif flag == 1:
        if x1 < x2:
            mask = (testpts >= x1) & (testpts < x2)
        else:
            mask = (testpts > x2) & (testpts <= x1)
    elif flag == 2:
        mask = (testpts >= lo) & (testpts <= hi)
    else:
        raise ValueError("flag must be 0, 1, or 2")

    indices = np.where(mask)[0]
    return indices
