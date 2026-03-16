"""vint2vrms.py – convert interval velocity to RMS velocity.

Mirrors the MATLAB ``vint2vrms`` from the Margrave / CREWES seismic toolbox.
"""

import numpy as np
from pwlint import pwlint


def vint2vrms(vint, t, tout=None):
    """
    Convert interval velocity to RMS velocity as a function of time.

    Parameters
    ----------
    vint : array_like
        Interval velocity vector (m/s).
    t : array_like
        Time vector (s) corresponding to *vint* (one-way or two-way,
        consistent throughout).
    tout : array_like, optional
        Output times at which Vrms is desired.  Must satisfy
        ``tout >= t[0]``.  Default: ``tout = t``.

    Returns
    -------
    vrms : np.ndarray
        RMS velocity at each time in *tout* (or *t* when *tout* is None).

    Notes
    -----
    * When *tout* is not supplied the integral is done directly on the input
      grid, matching the MATLAB ``cumsum`` path exactly.
    * When *tout* is supplied, a dense Vrms² curve is first computed on *t*
      and then interpolated to *tout* via :func:`pwlint`.
    * NaN values that can appear beyond the last input time are filled by
      constant extrapolation from the last valid sample (MATLAB behaviour).
    """
    vint = np.asarray(vint, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()

    if len(vint) != len(t):
        raise ValueError("vint and t must have the same length")

    if tout is None:
        # --- direct cumsum on the input grid ---
        nt = len(t)
        dt = np.diff(t)
        i1 = np.arange(nt - 1)

        vrms = np.zeros(nt)
        vrms[i1] = np.cumsum(dt * vint[i1] ** 2)
        # avoid divide-by-zero at t[0]
        denom = t[i1 + 1] - t[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            vrms[i1] = np.where(denom > 0, np.sqrt(vrms[i1] / denom), 0.0)
        vrms[-1] = vrms[-2]
        return vrms

    # --- arbitrary output times ---
    tout = np.asarray(tout, dtype=float).ravel()
    if np.any(tout < t[0]):
        raise ValueError("tout must be >= t[0]")

    nt = len(t)
    dt = np.diff(t)
    i1 = np.arange(nt - 1)

    vrms2 = np.zeros(nt)
    vrms2[i1] = np.cumsum(dt * vint[i1] ** 2)
    denom = t[i1 + 1] - t[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        vrms2[i1] = np.where(denom > 0, vrms2[i1] / denom, 0.0)
    vrms2[-1] = vrms2[-2]

    vrms_dense = np.sqrt(np.maximum(vrms2, 0.0))

    # Interpolate to tout
    vrms = pwlint(t, vrms_dense, tout)

    # Fill trailing NaNs by constant extrapolation (MATLAB behaviour)
    nan_idx = np.where(np.isnan(vrms))[0]
    if len(nan_idx) > 0:
        first_nan = nan_idx[0]
        if first_nan > 0:
            vrms[nan_idx] = vrms[first_nan - 1]
        else:
            vrms[nan_idx] = 0.0

    return vrms
