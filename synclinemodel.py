import numpy as np
from afd_vmodel import afd_vmodel


def synclinemodel(dx, xmax=2500, zmax=1000, vhigh=4000, vlow=2000,
                  zsyncline=None, radius=None, zfocal=None):
    """
    SYNCLINEMODEL: build a velocity model representing a syncline in a
    stratigraphic sequence.

    The syncline is modelled as a semi-circle connected to a horizontal
    interface::

                            x <== focal point of the syncline (zfocal)

        ------------*                 *--------------- <== zsyncline
                     *               *  <== syncline
                        *         *
                           *****

    By choosing the focal point below the surface (positive zfocal) you get a
    buried-focus reverse-time branch; choosing it above (negative zfocal) gives
    only a temporal syncline.

    Parameters
    ----------
    dx : float
        Grid interval (distance between grid points in x and z).
    xmax : float, optional
        Maximum x coordinate (minimum is 0). Default 2500.
    zmax : float, optional
        Maximum z coordinate (minimum is 0). Default 1000.
    vhigh : float, optional
        Velocity below the syncline. Default 4000.
    vlow : float, optional
        Velocity above the syncline. Default 2000.
    zsyncline : float, optional
        Depth to the horizontal interface surrounding the syncline.
        Default zmax / 3.
    radius : float, optional
        Radius of the semi-circle. Default zmax / 2.
        NOTE: zfocal + radius must exceed zsyncline or there is no syncline.
    zfocal : float, optional
        Depth to the focal point of the syncline (should not exceed zsyncline).
        Default zmax / 10.

    Returns
    -------
    vel : np.ndarray
        Velocity model matrix, shape (nz, nx).
    x : np.ndarray
        X coordinate vector.
    z : np.ndarray
        Z coordinate vector.

    Notes
    -----
    A simple way to visualise the result::

        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(vel, extent=[x[0], x[-1], z[-1], z[0]], aspect='auto',
                   cmap='seismic')
        plt.colorbar(label='Velocity (m/s)')
        plt.xlabel('X (m)')
        plt.ylabel('Z (m)')
        plt.title('Syncline velocity model')
        plt.show()
    """
    # Apply defaults that depend on other parameters
    if zsyncline is None:
        zsyncline = zmax / 3.0
    if radius is None:
        radius = zmax / 2.0
    if zfocal is None:
        zfocal = zmax / 10.0

    # Coordinate vectors
    x = np.arange(0, xmax + dx, dx)
    z = np.arange(0, zmax + dx, dx)

    # Flood with vhigh
    vel = np.ones((len(z), len(x))) * vhigh

    # ---- Install the flat layer above zsyncline ----
    x1 = x.min() - dx
    x2 = x.max() + dx
    z1 = -dx
    z2 = zsyncline
    xpoly = [x1, x2, x2, x1]
    zpoly = [z1, z1, z2, z2]
    vel = afd_vmodel(dx, vel, vlow, xpoly, zpoly)

    # ---- Install the syncline (semi-circle arc -> polygon) ----
    npoints = 500
    xf = xmax / 2.0          # x coordinate of focal point (centre)
    xpoly = np.linspace(xf - radius, xf + radius, npoints)
    zpoly = np.sqrt(np.maximum(radius**2 - (xf - xpoly)**2, 0.0)) + zfocal
    vel = afd_vmodel(dx, vel, vlow, xpoly, zpoly)

    return vel, x, z
