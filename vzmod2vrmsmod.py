"""vzmod2vrmsmod.py – convert interval velocity model V(x,z) to Vrms(x,t).

Mirrors the MATLAB ``vzmod2vrmsmod`` from the Margrave / CREWES toolbox.
"""

import numpy as np
from vint2t import vint2t
from vint2vrms import vint2vrms


def vzmod2vrmsmod(vel, z, dt, tmax, flag=1):
    """
    Convert an interval velocity model in depth to an RMS velocity model
    in time.

    Parameters
    ----------
    vel : np.ndarray, shape (nz, nx)
        Interval velocity model in depth.  Each row is a constant depth;
        each column is one lateral position.
    z : array_like, length nz
        Depth coordinate vector (m).  ``len(z)`` must equal ``vel.shape[0]``.
    dt : float
        Desired output time sample interval (s).
    tmax : float
        Maximum two-way time desired (s).
    flag : int, optional
        Controls behaviour when the model's maximum two-way traveltime is
        less than *tmax*:

        * ``1`` – extend the final Vrms value by constant extrapolation
          *(default)*.
        * ``2`` – extend the last interval velocity layer to *tmax* before
          computing Vrms (more physical but can yield a very fast half-space).

    Returns
    -------
    vrms : np.ndarray, shape (nt, nx)
        RMS velocity model in two-way time.
    t : np.ndarray, length nt
        Two-way time coordinate vector (s).

    Notes
    -----
    Progress is printed every 100 lateral positions, matching the MATLAB
    ``disp`` calls.
    """
    vel = np.asarray(vel, dtype=float)
    z = np.asarray(z, dtype=float).ravel()

    if len(z) != vel.shape[0]:
        raise ValueError("vel and z sizes are not compatible")

    nx = vel.shape[1]
    t = np.arange(0.0, tmax + dt / 2, dt)   # two-way time axis
    nt = len(t)
    vrms = np.zeros((nt, nx))

    for k in range(nx):
        # Two-way traveltime at the k-th lateral position
        tv_1way, _ = vint2t(vel[:, k], z)
        tv = 2.0 * tv_1way            # two-way times at each depth sample

        if flag == 2 and tv[-1] < tmax:
            # Extend the last interval layer to tmax
            tv[-1] = tmax

        vrms[:, k] = vint2vrms(vel[:, k], tv, t)

        if (k + 1) % 100 == 0:
            print(f"finished location {k+1} of {nx}")

    return vrms, t
