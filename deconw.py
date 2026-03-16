"""
deconw.py  –  Wiener (Levinson-Toeplitz) deconvolution.

Direct port of the CREWES MATLAB function deconw.m to Python/NumPy,
using the companion helper modules auto.py, levrec.py, and convm.py.

MATLAB source (Margrave, May 1991)
-----------------------------------
    function [trout, x] = deconw(trin, trdsign, n, stab)
    % stab default = 0.0001
    a = auto(trdsign, n, 0);   % n lags, no normalisation
    a(1) = a(1) * (1.0 + stab);
    b = [1.0  zeros(1, length(a)-1)];
    x = levrec(a, b);
    trout = convm(trin, x);
"""

import numpy as np
from auto import auto
from levrec import levrec
from convm import convm


def deconw(
    trin: np.ndarray,
    trdsign: np.ndarray,
    n: int,
    stab: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Wiener (Levinson-Toeplitz) deconvolution.

    Designs an *n*-point prediction-error (spiking) filter from *trdsign*
    using the Yule-Walker / Levinson equations and applies it to *trin*.

    Parameters
    ----------
    trin : array-like, 1-D
        Input trace to be deconvolved.
    trdsign : array-like, 1-D
        Trace used for operator design (often the same as *trin*).
    n : int
        Number of autocorrelation lags **and** length of the inverse
        operator.  Matches the MATLAB parameter of the same name.
    stab : float, optional
        Stabilisation (white-noise) factor expressed as a fraction of the
        zero-lag autocorrelation value.  Default = 1e-4 (MATLAB default).

    Returns
    -------
    trout : np.ndarray
        Deconvolved output trace, same length as *trin*.
    x : np.ndarray
        Deconvolution (inverse) operator, length *n*.

    Notes
    -----
    The estimated source wavelet can be recovered as::

        w = np.real(np.fft.ifft(1.0 / np.fft.fft(x, n=len(trin))))

    The operator is designed by solving the normal (Toeplitz) equations::

        T · x = e₀

    where T[i,j] = a[|i-j|] is the autocorrelation matrix, e₀ = [1, 0, …, 0]
    is a unit spike, and the solution *x* is the spiking deconvolution filter.
    """
    trin    = np.asarray(trin,    dtype=float).ravel()
    trdsign = np.asarray(trdsign, dtype=float).ravel()

    # ------------------------------------------------------------------
    # 1. Autocorrelation of the design trace
    #    MATLAB: a = auto(trdsign, n, 0)
    #      * n lags  →  output length n  (lags 0 … n-1)
    #      * flag=0  →  NO normalisation (raw energy-scaled values)
    # ------------------------------------------------------------------
    a = auto(trdsign, n, 0.0)          # shape (n,)

    # ------------------------------------------------------------------
    # 2. Stabilise zero-lag (add white noise)
    #    MATLAB: a(1) = a(1) * (1.0 + stab)
    # ------------------------------------------------------------------
    a[0] *= (1.0 + stab)

    # ------------------------------------------------------------------
    # 3. Right-hand side: unit spike at lag 0
    #    MATLAB: b = [1.0  zeros(1, length(a)-1)]
    # ------------------------------------------------------------------
    b      = np.zeros(n)
    b[0]   = 1.0

    # ------------------------------------------------------------------
    # 4. Levinson recursion → inverse operator
    #    MATLAB: x = levrec(a, b)
    #    Note: levrec normalises aa by max(aa) internally, so passing the
    #    raw (un-normalised) autocorrelation is correct.
    # ------------------------------------------------------------------
    x = levrec(a, b)                   # shape (n,)

    # ------------------------------------------------------------------
    # 5. Apply operator to input trace
    #    MATLAB: trout = convm(trin, x)
    # ------------------------------------------------------------------
    trout = convm(trin, x)             # shape (len(trin),)

    return trout, x
