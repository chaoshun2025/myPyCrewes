"""vrms2vint.py – convert RMS velocity to interval velocity.

Mirrors the MATLAB ``vrms2vint`` from the Margrave / CREWES seismic toolbox.
"""

import numpy as np
from interpextrap import interpextrap


def vrms2vint(vrms, t, flag=0):
    """
    Convert RMS velocity to interval velocity (Dix inversion).

    Parameters
    ----------
    vrms : array_like
        RMS velocity vector (m/s).
    t : array_like
        Time vector (s) corresponding to *vrms* (one-way or two-way).
    flag : int, optional
        Controls handling of non-physical (negative ``vint²``) results:

        * ``0`` – return NaN for non-physical values  *(default)*
        * ``1`` – replace non-physical values by linear interpolation /
          extrapolation from the neighbouring physical values.

    Returns
    -------
    vint : np.ndarray
        Interval velocity (m/s), same length as *vrms* / *t*.

    Notes
    -----
    The Dix formula used here is::

        vint²[i] = (vrms²[i+1]·(t[i+1]-t[0]) − vrms²[i]·(t[i]−t[0]))
                   / (t[i+1] − t[i])

    The last sample is filled by constant extrapolation from the
    second-to-last (matching MATLAB behaviour).
    Non-physical values occur when the numerator is negative, which can
    arise from noise or non-monotone Vrms curves.
    """
    vrms = np.asarray(vrms, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()

    nt = len(t)
    i1 = np.arange(nt - 1)
    i2 = np.arange(1, nt)

    vrms2 = vrms ** 2
    vint2 = np.zeros(nt)
    vint2[i1] = (vrms2[i2] * (t[i2] - t[0]) - vrms2[i1] * (t[i1] - t[0])) \
                / (t[i2] - t[i1])

    # Find and handle non-physical (negative) values
    bad = np.where(vint2 < 0)[0]
    if len(bad) > 0:
        if flag:
            # Interpolate / extrapolate from live (positive) samples
            live = np.where(vint2 > 0)[0]
            if len(live) > 0:
                vint2[bad] = interpextrap(t[live], vint2[live], t[bad])
            else:
                vint2[bad] = np.nan
        else:
            vint2[bad] = np.nan

    vint = np.sqrt(np.maximum(vint2, 0.0))
    # Propagate NaN where vint2 was NaN
    nan_mask = np.isnan(vint2)
    vint[nan_mask] = np.nan

    # Last sample: copy from second-to-last
    vint[-1] = vint[-2]

    return vint
