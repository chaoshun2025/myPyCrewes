import numpy as np


def findex(trin, flag=1.0):
    """
    FINDEX: returns indices of local maxima or minima in a trace.

    iex = findex(trin, flag)

    Note that extrema which persist for more than one sample will have only
    the *final* sample flagged (matching the MATLAB behaviour).

    Parameters
    ----------
    trin : array_like
        Input trace (1-D).
    flag : float, optional
         1.0  -> find local maxima.
         0.0  -> find both maxima and minima.
        -1.0  -> find local minima.
        Default = 1.0.

    Returns
    -------
    iex : np.ndarray of int
        0-based indices of the extrema in trin.
    """
    trin = np.asarray(trin, dtype=float).ravel()

    d1 = np.diff(trin)

    # keep only non-zero differences
    ind = np.where(d1 != 0)[0]
    d1 = d1[ind]
    d1 = d1 / np.abs(d1)   # +1 or -1

    d2 = np.diff(d1)        # -2 at max, +2 at min

    if flag > 0.0:
        iex_local = np.where(d2 < -1.9)[0]
    elif flag < 0.0:
        iex_local = np.where(d2 > 1.9)[0]
    else:
        iex_local = np.where(d2 != 0.0)[0]

    # +1 mirrors MATLAB iex=iex+1; then map back through ind
    iex_local = iex_local + 1
    iex = ind[iex_local]
    return iex
