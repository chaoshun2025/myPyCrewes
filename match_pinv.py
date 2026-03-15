"""
match_pinv.py  –  Pseudo-inverse match filter design.

Same as match.py but uses numpy.linalg.pinv (with a tolerance parameter)
instead of least-squares backslash (lstsq).  This gives the minimum-norm
solution when the convolution matrix is rank-deficient.

Port of match_pinv.m from the CREWES MATLAB toolbox.
"""

import numpy as np
from scipy.linalg import convolution_matrix


def match_pinv(
    trin: np.ndarray,
    trdsign: np.ndarray,
    t: np.ndarray,
    mlength: float,
    flag: int,
    tol: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Design a match filter using the pseudo-inverse (pinv) of the
    convolution matrix.

    Finds the filter ``mfilt`` of length *mlength* seconds that minimises
    ``||conv(mfilt, trin) - trdsign||²``, using pinv for stability.

    Parameters
    ----------
    trin : np.ndarray, 1D
        Input trace to be convolved with mfilt.
    trdsign : np.ndarray, 1D
        Target (desired) trace.  Same length as trin.
    t : np.ndarray, 1D
        Time coordinate vector for trin.
    mlength : float
        Desired filter length in seconds.
    flag : int
        0 → non-causal (zero-phase centred) operator.
        1 → causal operator (starts at t = 0).
    tol : float, optional
        Tolerance multiplier applied to the default pinv threshold
        ``max(size(TRIN)) * norm(TRIN) * eps``.
        Default = 1.0 (MATLAB default behaviour).

    Returns
    -------
    mfilt : np.ndarray, 1D
        Match filter of length ``round(mlength / dt) + 1``.
    tm : np.ndarray, 1D
        Time coordinate vector for mfilt.
    """
    trin    = np.asarray(trin,    dtype=float).ravel()
    trdsign = np.asarray(trdsign, dtype=float).ravel()
    t       = np.asarray(t,       dtype=float).ravel()

    dt = t[1] - t[0]
    n  = int(round(mlength / dt)) + 1

    # Build full convolution matrix  (len(trin)+n-1  ×  n)
    TRIN = convolution_matrix(trin, n, mode='full')   # shape (N+n-1, n)

    # Default tolerance (matches MATLAB's tol_default)
    tol_default = max(TRIN.shape) * np.linalg.norm(TRIN) * np.finfo(float).eps
    rcond = tol * tol_default / (np.linalg.norm(TRIN) + 1e-300)

    TRIN_pinv = np.linalg.pinv(TRIN, rcond=rcond)

    if flag == 1:
        # Causal operator
        rhs   = np.concatenate([trdsign, np.zeros(n - 1)])
        mfilt = TRIN_pinv @ rhs
        tm    = np.arange(n) * dt
    else:
        # Non-causal (centred) operator
        nh  = n // 2
        rhs = np.concatenate([
            np.zeros(nh),
            trdsign,
            np.zeros(n - nh - 1),
        ])
        mfilt = TRIN_pinv @ rhs
        tm    = np.arange(n) * dt - nh * dt

    return mfilt, tm
