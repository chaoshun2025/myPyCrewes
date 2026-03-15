import numpy as np


def balans(trin, trref=None, tz=None):
    """
    BALANS: match the rms power of one trace to another.

    trout = balans(trin, trref, tz)

    Adjusts the rms power of trin to equal that of trref.

    Parameters
    ----------
    trin : array_like
        Input trace (1-D) or gather (2-D, traces in columns).
    trref : array_like, optional
        Reference trace (1-D vector).
        Default = np.ones_like(trin) (for 1-D input).
    tz : array_like of int, optional
        Indices specifying the time zone over which balancing is done.
        Default = all samples shared by trin and trref.

    Returns
    -------
    trout : np.ndarray
        Balanced output, same shape as trin.

    Notes
    -----
    If trin is a 2-D matrix, traces are assumed to be in the columns.
    trref must be a 1-D vector.
    """
    trin = np.asarray(trin, dtype=float)

    if trref is None:
        trref = np.ones(trin.shape[0] if trin.ndim == 2 else len(trin))
    trref = np.asarray(trref, dtype=float).ravel()

    if trref.ndim != 1:
        raise ValueError("trref must be a vector")

    if tz is None:
        n = min(len(trref), trin.shape[0] if trin.ndim == 2 else len(trin))
        tz = np.arange(n)
    else:
        tz = np.asarray(tz, dtype=int)

    if trin.ndim == 1:
        denom = np.linalg.norm(trin[tz])
        if denom == 0:
            return trin.copy()
        trout = trin * np.linalg.norm(trref[tz]) / denom
    else:
        trout = np.zeros_like(trin)
        top = np.linalg.norm(trref[tz])
        for k in range(trin.shape[1]):
            denom = np.linalg.norm(trin[tz, k])
            if denom == 0:
                trout[:, k] = trin[:, k]
            else:
                trout[:, k] = trin[:, k] * top / denom
    return trout
