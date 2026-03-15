"""
deconw.py  –  Wiener (Levinson-Toeplitz) deconvolution.

Ports the CREWES MATLAB function deconw.m to Python/NumPy.
Helper utilities (auto, levrec, convm) are implemented inline.
"""

import numpy as np
from numpy.linalg import solve
import scipy.linalg as la


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto(x: np.ndarray, nlag: int, normalise: bool = False) -> np.ndarray:
    """
    Compute the one-sided autocorrelation of x up to nlag lags (inclusive).

    Parameters
    ----------
    x : 1D array
    nlag : int
        Number of lags (autocorrelation has nlag+1 values: lags 0..nlag).
    normalise : bool
        If True, normalise so that lag-0 == 1.

    Returns
    -------
    a : np.ndarray, shape (nlag+1,)
        a[k] = sum_t x[t] * x[t+k]
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    a = np.array([np.dot(x[:n - k], x[k:]) for k in range(nlag + 1)])
    if normalise and a[0] != 0:
        a /= a[0]
    return a


def _levrec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the Toeplitz system  T x = b  via the Levinson recursion,
    where T is symmetric Toeplitz with first column/row a.

    Parameters
    ----------
    a : 1D array, length n
        First row of the symmetric Toeplitz matrix (a[0] = diagonal).
    b : 1D array, length n
        Right-hand side.

    Returns
    -------
    x : 1D array, length n
    """
    # Use scipy's efficient Toeplitz solver
    T = la.toeplitz(a)
    return np.linalg.solve(T, b)


def _convm(trin: np.ndarray, filt: np.ndarray) -> np.ndarray:
    """
    Same-length convolution: returns the first len(trin) samples of the
    full linear convolution of trin with filt (causal / minimum-phase style).
    """
    n = len(trin)
    full = np.convolve(trin.astype(float), filt.astype(float))
    return full[:n]


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def deconw(
    trin: np.ndarray,
    trdsign: np.ndarray,
    n: int,
    stab: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Wiener (Levinson-Toeplitz) deconvolution.

    Designs an n-point prediction-error filter from trdsign using the
    Yule-Walker (Levinson) equations and applies it to trin.

    Parameters
    ----------
    trin : np.ndarray
        Input trace to be deconvolved (1D).
    trdsign : np.ndarray
        Trace used for operator design (1D).
    n : int
        Number of autocorrelation lags / length of inverse operator.
    stab : float, optional
        Stabilisation factor as a fraction of the zero-lag autocorrelation.
        Default = 1e-4.

    Returns
    -------
    trout : np.ndarray
        Deconvolved output trace (same length as trin).
    x : np.ndarray
        Deconvolution operator of length n.
        The estimated wavelet is ``np.real(np.fft.ifft(1.0 / np.fft.fft(x)))``.
    """
    trin = np.asarray(trin, dtype=float).ravel()
    trdsign = np.asarray(trdsign, dtype=float).ravel()

    # Build autocorrelation (n lags → n+1 values; MATLAB uses n as #lags)
    a = _auto(trdsign, n - 1)          # length n

    # Stabilise the zero-lag
    a[0] *= (1.0 + stab)

    # Right-hand side: spike at lag 0
    b = np.zeros(n)
    b[0] = 1.0

    # Levinson recursion → inverse operator
    x = _levrec(a, b)

    # Apply operator to input trace
    trout = _convm(trin, x)

    return trout, x
