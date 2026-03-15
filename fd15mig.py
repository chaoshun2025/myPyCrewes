"""
fd15mig.py  –  15-degree finite-difference time migration.

Ports the CREWES MATLAB function fd15mig.m to Python/NumPy.
"""

import numpy as np
import time


def fd15mig(
    aryin: np.ndarray,
    aryvel,
    t,
    x,
    dtau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    15-degree finite-difference time migration.

    Parameters
    ----------
    aryin : np.ndarray, shape (nsamp, ntr)
        Zero-offset data matrix (one trace per column).
    aryvel : scalar, 1D array, or 2D array
        Velocity information:
        - scalar  → constant velocity.
        - 1D array of length nsamp → RMS velocity function of time,
          applied uniformly across all traces.
        - 2D array of shape (nsamp, ntr) → spatially variant velocity.
    t : scalar or 1D array
        If scalar, the time sample interval in seconds.
        If array of length nsamp, the time coordinate vector.
    x : scalar or 1D array
        If scalar, the spatial sample interval (same units as velocity).
        If array of length ntr, the x coordinate vector.
    dtau : float
        Downward-continuation step length in **milliseconds**.

    Returns
    -------
    arymig : np.ndarray, shape (nsamp, ntr)
        Migrated time section.
    tmig : np.ndarray
        Time coordinate vector (same as input t vector).
    xmig : np.ndarray
        Spatial coordinate vector (same as input x vector).
    """
    t0 = time.time()

    aryin = np.asarray(aryin, dtype=float)
    nsamp, ntr = aryin.shape

    # ---- process t ----
    if np.ndim(t) > 0 and len(np.atleast_1d(t)) > 1:
        t = np.asarray(t, dtype=float).ravel()
        if len(t) != nsamp:
            raise ValueError("Length of t must equal number of rows in aryin.")
        dt = t[1] - t[0]
    else:
        dt = float(t)
        t = np.arange(nsamp) * dt

    # ---- process x ----
    if np.ndim(x) > 0 and len(np.atleast_1d(x)) > 1:
        x = np.asarray(x, dtype=float).ravel()
        if len(x) != ntr:
            raise ValueError("Length of x must equal number of columns in aryin.")
        dx = x[1] - x[0]
    else:
        dx = float(x)
        x = np.arange(ntr) * dx

    tmig = t.copy()
    xmig = x.copy()

    # ---- build velocity matrix ----
    aryvel = np.asarray(aryvel, dtype=float)
    if aryvel.ndim == 0 or aryvel.size == 1:
        vel_mat = float(aryvel) * np.ones((nsamp, ntr))
    elif aryvel.ndim == 1:
        v = aryvel.ravel()
        if len(v) == ntr and len(v) != nsamp:
            v = v.reshape(1, -1) * np.ones((nsamp, 1))
        elif len(v) == nsamp:
            v = v.reshape(-1, 1) * np.ones((1, ntr))
        else:
            raise ValueError("Velocity vector has wrong size.")
        vel_mat = v
    else:
        if aryvel.shape != (nsamp, ntr):
            raise ValueError("Velocity matrix must have shape (nsamp, ntr).")
        vel_mat = aryvel

    # ---- number of downward steps ----
    # dtau is in milliseconds → convert to seconds for comparison with dt
    dtau_s = dtau * 1e-3
    ktausm = max(1, int(np.ceil(dtau_s / dt)))
    ndown = nsamp // ktausm

    print(f"{ndown} steps required")

    arymig = aryin.copy()

    # ---- main loop ----
    for idown in range(1, ndown + 1):
        top1 = (idown - 1) * ktausm    # inclusive lower bound (0-based)
        top2 = min(idown * ktausm, nsamp - 1)

        aryn = np.zeros((2, ntr))
        aryo = np.zeros((2, ntr))

        for tlev in range(nsamp - 1, top1 - 1, -1):
            taueff_s = dtau_s
            if tlev < top2:
                taueff_s = dtau_s * (tlev - top1) / ktausm

            aryo[0, :] = arymig[tlev, :]

            # half velocity at the representative time level (0-based index)
            mid_idx = min(round((top1 + top2) / 2), nsamp - 1)
            vel = 0.5 * vel_mat[mid_idx, :]

            w = vel ** 2 * taueff_s * dt / (4.0 * dx ** 2)

            # second differences (interior points)
            new_diff = aryn[1, :-2] - 2.0 * aryn[1, 1:-1] + aryn[1, 2:]
            old_diff = aryo[0, :-2] - 2.0 * aryo[0, 1:-1] + aryo[0, 2:]

            aryn[0, 1:-1] = (
                w[1:-1] * (new_diff + old_diff)
                + aryo[0, 1:-1]
                + aryn[1, 1:-1]
                - aryo[1, 1:-1]
            )

            # boundary conditions (replicate nearest interior)
            aryn[0, 0] = aryn[0, 1]
            aryn[0, -1] = aryn[0, -2]

            aryn[1, :] = aryn[0, :]
            aryo[1, :] = aryo[0, :]

            arymig[tlev, :] = aryn[0, :]

        if idown % 10 == 0:
            print(f"Completed step {idown} of {ndown}")

    elapsed = time.time() - t0
    print(f"Migration completed in {elapsed:.1f} seconds")

    return arymig, tmig, xmig
