"""
del2_9pt.py — Nine-point (4th-order) Laplacian operator on a 2-D grid.

Translated from MATLAB del2_9pt.m (CREWES / Margrave).

The stencil is the standard 4th-order compact Laplacian in each direction::

    (-u[i-2] + 16*u[i-1] - 30*u[i] + 16*u[i+1] - u[i+2]) / (12 * delx^2)

applied independently to z and x, then summed.
"""

import numpy as np


def del2_9pt(u: np.ndarray, delx: float) -> np.ndarray:
    """
    Compute the 9-point approximation to the 2-D Laplacian of *u*.

    Boundaries are handled by zero-padding (Dirichlet BCs), matching the
    MATLAB original.

    Parameters
    ----------
    u    : ndarray, shape (nz, nx)
        2-D wavefield snapshot.
    delx : float
        Grid spacing (same in both directions).

    Returns
    -------
    output : ndarray, shape (nz, nx)
        Discrete Laplacian of *u*.
    """
    u    = np.asarray(u, dtype=float)
    nz, nx = u.shape
    factor = 1.0 / (12.0 * delx ** 2)

    # zero-pad by 2 on each side → (nz+4, nx+4)
    u2 = np.zeros((nz + 4, nx + 4))
    u2[2:nz + 2, 2:nx + 2] = u

    # ---- z-direction (rows) ------------------------------------------------
    lap_z = (-u2[0:nz,       2:nx + 2]
             + 16.0 * u2[1:nz + 1, 2:nx + 2]
             - 30.0 * u2[2:nz + 2, 2:nx + 2]
             + 16.0 * u2[3:nz + 3, 2:nx + 2]
             -        u2[4:nz + 4, 2:nx + 2]) * factor

    # ---- x-direction (columns) — operate on transposed array for clarity ---
    u2t = u2.T                          # (nx+4, nz+4)
    lap_x = (-u2t[0:nx,       2:nz + 2]
              + 16.0 * u2t[1:nx + 1, 2:nz + 2]
              - 30.0 * u2t[2:nx + 2, 2:nz + 2]
              + 16.0 * u2t[3:nx + 3, 2:nz + 2]
              -        u2t[4:nx + 4, 2:nz + 2]) * factor
    lap_x = lap_x.T                     # back to (nz, nx)

    return lap_z + lap_x
