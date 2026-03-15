"""
padpow2.py — Zero-pad a 1-D trace (or 2-D column matrix) to the next power of 2.

Translated from MATLAB padpow2.m (CREWES / Margrave).
"""

import numpy as np

def _next_pow2(n):
    """Return the smallest power of 2 >= n."""
    return 1 << int(np.ceil(np.log2(n))) if n > 1 else 1


def padpow2(trin: np.ndarray, flag: int = 0) -> np.ndarray:
    """
    Pad *trin* with zeros to the next power of two in the first (row) dimension.

    Parameters
    ----------
    trin : ndarray
        Input trace (column vector or 2-D array, samples along axis 0).
    flag : int, optional
        * ``0`` (default) — if the trace is already an exact power of two,
          leave its length unchanged.
        * ``1`` — if the trace is already an exact power of two, double it.

    Returns
    -------
    trout : ndarray
        Zero-padded trace / matrix (same dtype as *trin*).
    """
    trin = np.asarray(trin)
    n = trin.shape[0]

    #n2 = int(2 ** np.ceil(np.log2(n))) if n > 1 else 1
    n2 = _next_pow2(n)
    
    # if already a power of two and flag==1, double
    if flag == 1 and n2 == n:
        n2 = n * 2

    pad = n2 - n
    if trin.ndim == 1:
        return np.concatenate([trin, np.zeros(pad, dtype=trin.dtype)])
    else:
        pad_block = np.zeros((pad, trin.shape[1]), dtype=trin.dtype)
        return np.vstack([trin, pad_block])
