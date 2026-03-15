"""
del2_5pt.py — Five-point Laplacian operator on a 2-D grid.

Translated from MATLAB del2_5pt.m (CREWES / Margrave).

The grid spacing must be identical in both the horizontal (x) and vertical
(z) directions.
"""

import numpy as np


def del2_5pt(u: np.ndarray, delx: float) -> np.ndarray:
    """
    Compute the 5-point approximation to the 2-D Laplacian of *u*.

    The stencil is the standard cross-shaped 5-point operator::

        (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j]) / delx^2

    Boundaries are handled by zero-padding (equivalent to Dirichlet BCs,
    matching the MATLAB original).

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
    factor = 1.0 / delx ** 2

    # zero-pad by 1 on each side → (nz+2, nx+2)
    u2 = np.zeros((nz + 2, nx + 2))
    u2[1:nz + 1, 1:nx + 1] = u

    # 5-point stencil  (row direction = z, col direction = x)
    lap = (u2[2:nz + 2, 1:nx + 1]   # z+1
         + u2[0:nz,     1:nx + 1]   # z-1
         + u2[1:nz + 1, 2:nx + 2]   # x+1
         + u2[1:nz + 1, 0:nx]       # x-1
         - 4.0 * u2[1:nz + 1, 1:nx + 1]) * factor

    return lap
