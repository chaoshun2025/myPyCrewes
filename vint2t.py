"""vint2t.py – compute one-way vertical travel-time from interval velocity.

Mirrors the MATLAB ``vint2t`` from the Margrave / CREWES seismic toolbox.
"""

import numpy as np
from pcint import pcint


def vint2t(vint, z, zout=None, tnot=None):
    """
    Compute 1-way vertical travel-time given interval velocity vs depth.

    Parameters
    ----------
    vint : array_like
        Interval velocity vector (m/s).  ``vint[k]`` is the velocity between
        ``z[k]`` and ``z[k+1]``; the last value is used as a half-space.
    z : array_like
        Depth vector (m), same length as *vint*.
    zout : array_like, optional
        Depths at which output travel-times ``t1`` are desired.
        Default: ``zout = z``  (returns ``t1 == t2``).
    tnot : float, optional
        Constant time shift (s) – the 1-way time from the surface to
        ``z[0]``.  Default: ``z[0] / vint[0]``.

    Returns
    -------
    t1 : np.ndarray
        One-way vertical times (s) at *zout* (or *z* when *zout* is None).
    t2 : np.ndarray
        One-way vertical times (s) at *z*.
        When *zout* is None, ``t2 == t1``.

    Notes
    -----
    MATLAB usage ``t1=vint2t(vint,z)`` corresponds to unpacking just the
    first return value: ``t1, _ = vint2t(vint, z)``.

    The MATLAB version returns a vector of length ``nz+1`` when called
    without *zout* (because it prepends t=0 at the surface as an extra
    element). This Python version matches that behaviour: the first element
    is ``tnot`` and subsequent elements are cumulative integrals, giving a
    vector of length ``nz+1`` where ``nz = len(z)``. Specifically::

        z  = dz*(0:nz-1)   →  t has length nz  (same as z)

    when called from ``vz2vt`` as ``t1 = vint2t(vel[:,s], z)`` where
    ``z = dz*(0:vz)`` has length ``vz+1``.

    To be consistent with vz2vt which passes ``z = dz*(0:vz)`` (length
    vz+1) and vel of shape (vz, vx), this function handles the case where
    len(z) == len(vint)+1 by treating vint as piecewise-constant over the
    depth intervals defined by z.
    """
    vint = np.asarray(vint, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()

    # Handle the case where z has one more element than vint (interval style)
    if len(z) == len(vint) + 1:
        # vint[k] is the velocity in interval [z[k], z[k+1]]
        if tnot is None:
            tnot = 0.0
        dz = np.diff(z)
        t = np.zeros(len(z))
        t[1:] = np.cumsum(dz / vint)
        t += tnot
        if zout is None:
            return t, t.copy()
        # Interpolate to zout
        zout = np.asarray(zout, dtype=float).ravel()
        t1 = np.interp(zout, z, t)
        return t1, t

    if len(vint) != len(z):
        raise ValueError(
            "vint and z must be vectors of the same length "
            "(or z may have one more element than vint)"
        )

    if tnot is None:
        tnot = z[0] / vint[0] if z[0] != 0.0 else 0.0

    if zout is None:
        # Simple case: output on the same grid as z
        nz = len(z)
        dz = np.diff(z)
        t = np.zeros(nz)
        t[1:] = np.cumsum(dz / vint[:nz - 1])
        t += tnot
        return t, t.copy()

    # General case: arbitrary output depths
    zout = np.asarray(zout, dtype=float).ravel()
    vintout = pcint(z, vint, zout)

    nz = len(z)
    znew = np.concatenate([z, zout])
    vnew = np.concatenate([vint, vintout])
    isort = np.argsort(znew, kind="stable")
    znew_s = znew[isort]
    vnew_s = vnew[isort]

    nz2 = len(znew_s)
    dz2 = np.diff(znew_s)
    tnew = np.zeros(nz2)
    tnew[1:] = np.cumsum(dz2 / vnew_s[:nz2 - 1])
    tnew += tnot

    # Unsort
    t_unsorted = np.empty_like(tnew)
    t_unsorted[isort] = tnew

    t2 = t_unsorted[:nz]
    t1 = t_unsorted[nz:]
    return t1, t2
