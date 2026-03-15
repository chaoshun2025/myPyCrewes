import numpy as np
from matplotlib.path import Path


def afd_vmodel(dx, vmodin, vel, xpoly, zpoly):
    """
    AFD_VMODEL: makes simple polygonal velocity models.

    Superimposes a polygon with a different velocity onto the background
    velocity model.

    Parameters
    ----------
    dx : float
        Bin spacing for both horizontal and vertical (consistent units).
    vmodin : np.ndarray
        Background velocity matrix, shape (nz, nx). Upper-left corner is (0, 0).
    vel : float
        Velocity inside the polygon (scalar, consistent units).
    xpoly : array-like
        X coordinates of the polygon vertices (consistent units).
    zpoly : array-like
        Z coordinates of the polygon vertices (consistent units).

    Returns
    -------
    vmodout : np.ndarray
        Velocity matrix with the polygon superimposed.
    """
    xpoly = np.asarray(xpoly)
    zpoly = np.asarray(zpoly)

    if xpoly.shape != zpoly.shape:
        raise ValueError("xpoly must be the same size as zpoly")
    if np.isscalar(vel) is False and np.asarray(vel).size != 1:
        raise ValueError("vel must be a scalar")

    vmodout = vmodin.copy()
    nz, nx = vmodin.shape

    x = np.arange(nx) * dx          # shape (nx,)
    z = np.arange(nz) * dx          # shape (nz,)

    # Build grid of (x, z) sample points  shape (nz*nx, 2)
    xx, zz = np.meshgrid(x, z)      # each (nz, nx)
    pts = np.column_stack([xx.ravel(), zz.ravel()])

    # Build closed polygon path (matplotlib Path handles the closing)
    poly_pts = np.column_stack([xpoly, zpoly])
    path = Path(poly_pts)

    inside = path.contains_points(pts)          # (nz*nx,) bool
    inside = inside.reshape(nz, nx)

    vmodout[inside] = vel
    return vmodout
