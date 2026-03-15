"""
gaussian_smoother.py  –  Smooth a 2-D velocity model with a Gaussian kernel.

Smoothing is performed in slowness (1/v) space and then inverted, which
approximately conserves ray-theoretical travel time.

Port of gaussian_smoother.m (note: the MATLAB source has a typo in the
function name – 'gaussian_smooother' with three o's; the Python version
uses the correctly-spelled name).
"""

import numpy as np
from scipy.signal import fftconvolve


def gaussian_smoother(
    vel: np.ndarray,
    x,
    z,
    hw: float,
) -> np.ndarray:
    """
    Smooth a 2-D velocity model by convolving its slowness with a 2-D
    Gaussian and inverting the result.

    Parameters
    ----------
    vel : np.ndarray, shape (nz, nx)
        Input velocity model.
    x : scalar or 1D array
        Horizontal coordinates.  If scalar, interpreted as the grid
        spacing dx; if array its length must equal nx (= vel.shape[1]).
    z : scalar or 1D array
        Depth coordinates.  If scalar, interpreted as grid spacing dz;
        if array its length must equal nz (= vel.shape[0]).
    hw : float
        Half-width of the Gaussian smoother (same spatial units as x/z).
        The Gaussian is truncated at 3 * hw (captures >99.9 % of the area).

    Returns
    -------
    velsmo : np.ndarray, shape (nz, nx)
        Smoothed velocity model (same size as vel).
    """
    vel = np.asarray(vel, dtype=float)
    nz, nx = vel.shape

    # ---- x axis -----------------------------------------------------------
    x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    if x.size == 1:
        dx = float(x[0])
        x  = np.arange(nx) * dx
    else:
        if x.size != nx:
            raise ValueError("Length of x must equal vel.shape[1] (nx).")
        dx = x[1] - x[0]

    # ---- z axis -----------------------------------------------------------
    z = np.atleast_1d(np.asarray(z, dtype=float)).ravel()
    if z.size == 1:
        dz = float(z[0])
        z  = np.arange(nz) * dz
    else:
        if z.size != nz:
            raise ValueError("Length of z must equal vel.shape[0] (nz).")
        dz = z[1] - z[0]

    # ---- build 2-D Gaussian kernel ----------------------------------------
    # Truncate at 3 * hw in both directions
    ngx2 = int(round(3.0 * hw / dx))
    ngz2 = int(round(3.0 * hw / dz))

    xg = np.arange(-ngx2, ngx2 + 1) * dx           # shape (2*ngx2+1,)
    zg = np.arange(-ngz2, ngz2 + 1) * dz           # shape (2*ngz2+1,)

    XG, ZG = np.meshgrid(xg, zg)                    # (2*ngz2+1, 2*ngx2+1)
    sq_dist = XG ** 2 + ZG ** 2
    gauss   = np.exp(-sq_dist / (hw * hw))

    # ---- extend the velocity model (constant extrapolation) ---------------
    # Horizontal padding
    vel_ext = np.hstack([
        vel[:, :1]  * np.ones((nz, ngx2)),
        vel,
        vel[:, -1:] * np.ones((nz, ngx2)),
    ])
    # Vertical padding
    vel_ext = np.vstack([
        vel_ext[:1, :]  * np.ones((ngz2, vel_ext.shape[1])),
        vel_ext,
        vel_ext[-1:, :] * np.ones((ngz2, vel_ext.shape[1])),
    ])

    # ---- smooth slowness and invert ---------------------------------------
    slowness_ext = 1.0 / vel_ext
    gauss_norm   = gauss / gauss.sum()

    slow_smo = fftconvolve(slowness_ext, gauss_norm, mode='same')

    # Trim back to original size
    slow_smo_trimmed = slow_smo[ngz2: ngz2 + nz, ngx2: ngx2 + nx]

    velsmo = 1.0 / slow_smo_trimmed
    return velsmo
