"""
trend.py — Estimate the trend of a signal by low-order polynomial fit.

Translated from MATLAB trend.m (CREWES / Margrave).
"""

import numpy as np


def trend(s: np.ndarray,
          t: np.ndarray,
          m: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the trend of signal *s* via a degree-*m* polynomial fit.

    Parameters
    ----------
    s : ndarray, shape (n,)
        Input signal.
    t : ndarray, shape (n,)
        Time coordinate vector for *s*.
    m : int, optional
        Degree of polynomial fit (default 1, i.e. linear).

    Returns
    -------
    st : ndarray, shape (n,)
        Trend (polynomial evaluated at *t*).
    p  : ndarray
        Polynomial coefficients so that ``st = np.polyval(p, t_norm)``.
        ``t_norm`` is *t* divided by its mean, matching the MATLAB code.
    """
    s = np.asarray(s, dtype=float)
    t = np.asarray(t, dtype=float)

    # normalise t (mirrors MATLAB: t = t / mean(t))
    t_mean = np.mean(t)
    if t_mean != 0.0:
        t_norm = t / t_mean
    else:
        t_norm = t.copy()

    # clamp polynomial degree
    m = min(m, len(s) - 1)

    if m > 0:
        p  = np.polyfit(t_norm, s, m)
        st = np.polyval(p, t_norm)
    else:
        p  = np.array([0.0])
        st = np.zeros_like(s)

    return st, p
