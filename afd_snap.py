"""
afd_snap.py — Advance the acoustic wavefield by one time step.

Translated from MATLAB afd_snap.m (CREWES / Margrave).
"""

import numpy as np
from del2_5pt     import del2_5pt
from del2_9pt     import del2_9pt
from afd_bc_inner import afd_bc_inner
from afd_bc_outer import afd_bc_outer


def afd_snap(delx:      float,
             delt:      float,
             velocity:  np.ndarray,
             snap1:     np.ndarray,
             snap2:     np.ndarray,
             laplacian: int,
             boundary:  int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Propagate the acoustic wavefield forward by one time step.

    Uses the explicit second-order finite-difference scheme::

        snapshot = v² · dt² · ∇²(snap2) + 2·snap2 − snap1

    with absorbing boundary conditions applied afterwards.

    Parameters
    ----------
    delx      : float
        Spatial grid spacing (m) — identical in x and z.
    delt      : float
        Time step (s).
    velocity  : ndarray, shape (nz, nx)
        Velocity model (already halved for exploding-reflector convention).
    snap1     : ndarray, shape (nz, nx)
        Wavefield at t − delt.
    snap2     : ndarray, shape (nz, nx)
        Wavefield at t.
    laplacian : int
        ``1`` — 5-point (2nd-order) Laplacian.
        ``2`` — 9-point (4th-order) Laplacian.
    boundary  : int
        ``0`` — no absorbing boundaries.
        ``1`` — all four sides absorbing.
        ``2`` — three sides absorbing, top free.

    Returns
    -------
    snapshot : ndarray, shape (nz, nx)
        Wavefield at t + delt.
    z        : ndarray, shape (nz,)
        Depth coordinate vector (m).
    x        : ndarray, shape (nx,)
        Horizontal coordinate vector (m).
    """
    snap1    = np.asarray(snap1,    dtype=float)
    snap2    = np.asarray(snap2,    dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    nz, nx   = snap1.shape

    x = np.arange(nx) * delx
    z = np.arange(nz) * delx

    if laplacian == 2:
        # ---- 9-point Laplacian -----------------------------------------------
        snapshot = velocity**2 * delt**2 * del2_9pt(snap2, delx) + 2*snap2 - snap1

        # zero outer 2 rows/cols before applying BCs
        if boundary == 1:
            snapshot[:2,  :]  = 0.0
            snapshot[-2:, :]  = 0.0
            snapshot[:,  :2]  = 0.0
            snapshot[:, -2:]  = 0.0
        else:
            snapshot[-2:, :]  = 0.0
            snapshot[:,  :2]  = 0.0
            snapshot[:, -2:]  = 0.0

        if boundary:
            snapshot = afd_bc_inner(delx, delt, velocity,
                                    snap1, snap2, snapshot, boundary)
            snapshot = afd_bc_outer(delx, delt, velocity,
                                    snap1, snap2, snapshot, boundary)

    else:
        # ---- 5-point Laplacian -----------------------------------------------
        snapshot = velocity**2 * delt**2 * del2_5pt(snap2, delx) + 2*snap2 - snap1

        # zero outer 1 row/col before applying BCs
        if boundary == 1:
            snapshot[0,  :]  = 0.0
            snapshot[-1, :]  = 0.0
            snapshot[:,  0]  = 0.0
            snapshot[:, -1]  = 0.0
        else:
            snapshot[-1, :]  = 0.0
            snapshot[:,  0]  = 0.0
            snapshot[:, -1]  = 0.0

        if boundary:
            snapshot = afd_bc_outer(delx, delt, velocity,
                                    snap1, snap2, snapshot, boundary)

    return snapshot, z, x
