"""
deconb.py  –  Burg-scheme (maximum entropy) deconvolution.

Depends on:
    burgpr  – Burg prediction-error filter (from your existing utils)
    convm   – minimum-phase (causal) convolution helper
    balans  – energy-balance scalar

These three helpers are implemented inline here so the file is
self-contained, but you can swap them out for your own versions.
"""

import numpy as np
from numpy.linalg import solve


# ---------------------------------------------------------------------------
# Helpers (self-contained equivalents of CREWES utilities)
# ---------------------------------------------------------------------------

def _burgpr(x: np.ndarray, order: int) -> np.ndarray:
    """
    Estimate a prediction-error filter (PEF) of given order using the
    Burg algorithm.

    Parameters
    ----------
    x : 1D np.ndarray
        Input signal used for operator design.
    order : int
        Filter order (= length of returned PEF, including the leading 1).

    Returns
    -------
    pef : 1D np.ndarray, shape (order,)
        Prediction-error filter with pef[0] == 1.0 (causal, minimum-phase).
    """
    x = x.astype(float)
    n = len(x)
    ef = x.copy()          # forward residual
    eb = x.copy()          # backward residual
    a = np.zeros(order)
    a[0] = 1.0

    for m in range(1, order):
        # Burg reflection coefficient
        num = -2.0 * np.dot(ef[m:], eb[m - 1: n - 1])
        den = np.dot(ef[m:], ef[m:]) + np.dot(eb[m - 1: n - 1], eb[m - 1: n - 1])
        if den == 0:
            km = 0.0
        else:
            km = num / den

        # Update filter coefficients
        a_new = a.copy()
        for k in range(1, m + 1):
            a_new[k] = a[k] + km * a[m - k]
        a = a_new

        # Update residuals
        ef_new = ef[m:] + km * eb[m - 1: n - 1]
        eb_new = eb[m - 1: n - 1] + km * ef[m:]
        ef[m:] = ef_new
        eb[m - 1: n - 1] = eb_new

    return a


def _convm(trin: np.ndarray, filt: np.ndarray) -> np.ndarray:
    """
    Minimum-phase (same-length) convolution: output has the same length
    as trin, taking only the first len(trin) samples of the full convolution.
    """
    n = len(trin)
    full = np.convolve(trin.astype(float), filt.astype(float))
    return full[:n]


def _balans(trout: np.ndarray, trin: np.ndarray) -> np.ndarray:
    """
    Scale trout so that its RMS energy matches that of trin.
    """
    e_in = np.sqrt(np.mean(trin ** 2))
    e_out = np.sqrt(np.mean(trout ** 2))
    if e_out == 0:
        return trout
    return trout * (e_in / e_out)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def deconb(
    trin: np.ndarray,
    trdsign: np.ndarray,
    l: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Burg-scheme deconvolution of an input trace.

    Parameters
    ----------
    trin : np.ndarray
        Input trace to be deconvolved (1D).
    trdsign : np.ndarray
        Trace used for operator design (1D, same length as trin).
    l : int
        Prediction-error filter length (= length of inverse operator).

    Returns
    -------
    trout : np.ndarray
        Deconvolved output trace (same length as trin).
    pefilt : np.ndarray
        The prediction-error (inverse) operator of length l.
    """
    trin = np.asarray(trin, dtype=float).ravel()
    trdsign = np.asarray(trdsign, dtype=float).ravel()

    # Estimate the PEF via Burg's method
    pefilt = _burgpr(trdsign, l)        # shape (l,)

    # Convolve PEF with input trace (same-length)
    trout = _convm(trin, pefilt)

    # Balance energy
    trout = _balans(trout, trin)

    return trout, pefilt
