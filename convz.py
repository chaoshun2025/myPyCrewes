"""
convz.py — Convolution then truncation for non-minimum-phase wavelets.

Translated from MATLAB convz.m (CREWES / Margrave).
"""

import numpy as np
from mwindow import mwindow


def convz(r:     np.ndarray,
          w:     np.ndarray,
          nzero: int  | None = None,
          nout:  int  | None = None,
          flag:  int         = 1,
          pct:   float       = 5.0) -> np.ndarray:
    """
    Convolve reflectivity *r* with wavelet *w*, then truncate the result.

    Designed for zero-phase (non-causal) wavelets: the sample at index
    *nzero* in *w* corresponds to zero-lag, so the output is time-aligned
    correctly.

    Parameters
    ----------
    r     : ndarray, shape (nsamps,) or (nsamps, ntr)
        Input reflectivity / trace(s).
    w     : ndarray, shape (nw,)
        Wavelet.
    nzero : int, optional
        1-based index of the zero-time sample in *w*.
        Defaults to ``ceil((len(w) + 1) / 2)``.
    nout  : int, optional
        Length of the output trace.  Defaults to ``nsamps``.
    flag  : int, optional
        ``1`` (default) — apply a cosine taper (mwindow) to the output.
        ``0`` — no taper.
    pct   : float, optional
        Percentage taper applied at both ends (used by mwindow).
        Default 5 %.

    Returns
    -------
    s : ndarray, same shape as *r*
        Convolved and truncated trace(s).
    """
    r = np.asarray(r, dtype=float)
    w = np.asarray(w, dtype=float).ravel()

    # ensure 2-D (samples × traces)
    transpose = False
    if r.ndim == 1:
        r = r[:, np.newaxis]
        transpose = True
    nsamps, ntr = r.shape

    if nzero is None:
        nzero = int(np.ceil((len(w) + 1) / 2))   # 1-based
    if nout is None:
        nout = nsamps

    win = mwindow(nout, pct) if flag == 1 else np.ones(nout)

    s = np.zeros((nout, ntr))
    for k in range(ntr):
        temp      = np.convolve(r[:, k], w)
        # nzero is 1-based; Python slice is 0-based
        s[:, k]   = temp[nzero - 1: nout + nzero - 1] * win

    if transpose:
        s = s[:, 0]
    return s
