"""
match.py  –  Least-squares match filter design.

Ports the CREWES MATLAB function match.m to Python/NumPy.
"""

import numpy as np
from numpy.linalg import lstsq


def match(
    trin: np.ndarray,
    trdsign: np.ndarray,
    t: np.ndarray,
    mlength: float,
    flag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Design a match filter that matches *trin* to *trdsign* in the
    least-squares sense.

    Solves:  min  ||conv(mfilt, trin) - trdsign||²

    Parameters
    ----------
    trin : np.ndarray, 1D
        Input trace to be matched to trdsign.
    trdsign : np.ndarray, 1D
        Target trace.  Must be the same length as trin.
    t : np.ndarray, 1D
        Time coordinate vector for trin.
    mlength : float
        Desired match filter length in seconds.
    flag : int
        0 → non-causal (zero-phase) operator (operator is centred).
        1 → causal operator.

    Returns
    -------
    mfilt : np.ndarray, 1D
        Match filter of length n = round(mlength / dt) + 1.
    tm : np.ndarray, 1D
        Time coordinate vector for mfilt.
    """
    trin    = np.asarray(trin,    dtype=float).ravel()
    trdsign = np.asarray(trdsign, dtype=float).ravel()
    t       = np.asarray(t,       dtype=float).ravel()

    dt = t[1] - t[0]
    n  = int(round(mlength / dt)) + 1

    # Build the convolution matrix  (Toeplitz structure)
    # TRIN has shape  (len(trin) + n - 1,  n)
    # such that TRIN @ mfilt  ≈  conv(trin, mfilt) [full]
    from scipy.linalg import convolution_matrix
    TRIN = convolution_matrix(trin, n, mode='full')   # (len(trin)+n-1, n)

    if flag == 1:
        # Causal operator: solve TRIN @ mfilt ≈ [trdsign; zeros(n-1)]
        rhs = np.concatenate([trdsign, np.zeros(n - 1)])
        mfilt, _, _, _ = lstsq(TRIN, rhs, rcond=None)
        tm = np.arange(n) * dt                        # starts at t=0
    else:
        # Non-causal (centred) operator
        nh = n // 2
        rhs = np.concatenate([
            np.zeros(nh),
            trdsign,
            np.zeros(n - nh - 1)
        ])
        mfilt, _, _, _ = lstsq(TRIN, rhs, rcond=None)
        tm = np.arange(n) * dt - nh * dt             # centred at t=0

    return mfilt, tm
