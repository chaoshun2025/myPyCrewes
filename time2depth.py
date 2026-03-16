"""time2depth.py – convert a seismic trace from time to depth.

Mirrors the MATLAB ``time2depth`` from the Margrave / CREWES seismic toolbox.
"""

import numpy as np
from pwlint import pwlint


def time2depth(trc, t, tzcurve, dz):
    """
    Convert a single seismic trace from time to depth by a simple stretch.

    The conversion is specified by a time-depth curve.  Between knot points
    linear interpolation is used.  Sinc (band-limited) interpolation is
    applied along the time axis.

    Parameters
    ----------
    trc : array_like
        Input trace samples in time.
    t : array_like
        Time coordinate vector for *trc* (regularly sampled).
    tzcurve : array_like, shape (n, 2)
        Time-depth curve.  First column: times (same units as *t*).
        Second column: corresponding depths (m).  The first time must be
        zero; the last time must be >= ``max(t)``.  Depths must increase
        monotonically.
    dz : float
        Output depth sample interval (m).

    Returns
    -------
    ztrc : np.ndarray
        Trace resampled to depth.
    z : np.ndarray
        Depth coordinate vector (m).

    Notes
    -----
    * To avoid aliasing, choose ``dz < vmin * dt / n`` where n=1 for
      one-way time, n=2 for two-way time, and vmin is the minimum velocity.
    * Sinc interpolation is performed via :func:`_sinci` (a pure-NumPy
      implementation).  For production use consider ``scipy.signal.resample``
      or a dedicated sinc-interpolation library.
    """
    trc = np.asarray(trc, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    tzcurve = np.asarray(tzcurve, dtype=float)

    # Validate regular sampling
    diffs = np.diff(t)
    if np.any(np.abs(np.diff(diffs)) > 1e-10):
        raise ValueError("Input time trace must be regularly sampled")

    tz = tzcurve[:, 0]
    zt = tzcurve[:, 1]

    if tz[0] != 0.0:
        raise ValueError("tz curve must start at zero time")
    if tz[-1] < t[-1]:
        raise ValueError("tz curve must extend to times >= max(t)")

    # Monotonic depth check
    if np.any(np.diff(zt) <= 0):
        raise ValueError("Depths on tzcurve must increase monotonically")

    # Depth range corresponding to the input time range
    z1 = pwlint(tz, zt, np.array([t[0]]))[0]
    z2 = pwlint(tz, zt, np.array([t[-1]]))[0]

    nz = int(round((z2 - z1) / dz))
    z = (np.arange(nz + 1) * dz)

    # Times at which we need the trace (map each depth to its time)
    tint = pwlint(zt, tz, z)

    # Sinc interpolation
    ztrc = _sinci(trc, t, tint)
    return ztrc, z


# ---------------------------------------------------------------------------
# Sinc (band-limited) interpolation  – mirrors MATLAB sinci
# ---------------------------------------------------------------------------

def _sinci(s, t, ti, lp=8):
    """
    Band-limited sinc interpolation.

    Parameters
    ----------
    s : np.ndarray
        Input signal samples.
    t : np.ndarray
        Time coordinate for *s* (regularly spaced).
    ti : np.ndarray
        Output times at which the signal is desired.
    lp : int, optional
        Half-length of the sinc operator (number of samples on each side).
        Default 8.

    Returns
    -------
    si : np.ndarray
        Interpolated signal at *ti*.
    """
    s = np.asarray(s, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    ti = np.asarray(ti, dtype=float).ravel()

    dt = t[1] - t[0]
    n = len(s)
    ni = len(ti)
    si = np.zeros(ni)

    for k in range(ni):
        # Fractional index
        idx = (ti[k] - t[0]) / dt
        j0 = int(np.floor(idx))
        # Sinc kernel over ±lp samples
        js = np.arange(j0 - lp + 1, j0 + lp + 1)
        valid = (js >= 0) & (js < n)
        js_v = js[valid]
        x = idx - js_v                        # fractional offset
        # Windowed sinc (Hanning window)
        w = np.sinc(x) * (0.5 + 0.5 * np.cos(np.pi * x / lp))
        si[k] = np.dot(s[js_v], w)

    return si
