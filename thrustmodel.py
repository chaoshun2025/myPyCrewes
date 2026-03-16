"""
thrustmodel.py  –  Build a velocity model representing a thrust sheet.

Direct port of Margrave's ``thrustmodel.m`` (CREWES toolbox).

Dependency
----------
afd_vmodel  –  ``afd_vmodel.py``  (polygonal velocity model builder)
"""

import numpy as np
from afd_vmodel import afd_vmodel


def thrustmodel(
    dx: float,
    xmax: float = 5100.0,
    zmax: float = 2500.0,
    vhigh: float = 3145.0,
    vlow: float = 2500.0,
    vbasement: float = 4000.0,
    targetflag: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a velocity matrix representing a thrust sheet within a lower-
    velocity background material.

    Parameters
    ----------
    dx : float
        Grid interval (distance between grid points in both x and z, metres).
    xmax : float, optional
        Maximum x coordinate (minimum is 0).  Default 5100.
    zmax : float, optional
        Maximum z coordinate (minimum is 0).  Default 2500.
    vhigh : float, optional
        Velocity inside the thrust sheet.  Default 3145.
    vlow : float, optional
        Velocity of the surrounding (background) material.  Default 2500.
    vbasement : float, optional
        Velocity of the basement.  Default 4000.
    targetflag : int, optional
        If 1, insert a small sub-thrust structure (the "target").
        Default 1.

    Returns
    -------
    vel : np.ndarray, shape (nz, nx)
        Velocity model matrix.
    x : np.ndarray, shape (nx,)
        X coordinate vector (metres).
    z : np.ndarray, shape (nz,)
        Z (depth) coordinate vector (metres).

    Notes
    -----
    The simplest way to view the model is::

        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(vel, extent=[x[0], x[-1], z[-1], z[0]],
                   aspect='auto', cmap='seismic')
        plt.colorbar(label='velocity (m/s)')
        plt.title('Thrust model – colours indicate velocity')
        plt.xlabel('distance (m)')
        plt.ylabel('depth (m)')
        plt.tight_layout()
        plt.show()
    """
    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------
    nx = int(np.floor(xmax / dx)) + 1
    nz = int(np.floor(zmax / dx)) + 1
    x  = np.arange(nx) * dx          # shape (nx,)
    z  = np.arange(nz) * dx          # shape (nz,)

    # ------------------------------------------------------------------
    # Flood model with background velocity (vlow)
    # ------------------------------------------------------------------
    vel = vlow * np.ones((nz, nx), dtype=float)

    # ------------------------------------------------------------------
    # poly1 – basement
    # ------------------------------------------------------------------
    xpoly = [-dx,    1170.0, xmax + dx, xmax + dx, -dx      ]
    zpoly = [1940.0, 1940.0, 1940.0,    zmax + dx,  zmax + dx]
    vel = afd_vmodel(dx, vel, vbasement, xpoly, zpoly)

    # ------------------------------------------------------------------
    # poly2 – thrust sheet block 1 (left wedge)
    # ------------------------------------------------------------------
    xpoly = [-dx,    990.0,      1170.0,      -dx      ]
    zpoly = [1410.0, 1410.0,     1940.0 + dx, 1940.0 + dx]
    vel = afd_vmodel(dx, vel, vhigh, xpoly, zpoly)

    # ------------------------------------------------------------------
    # poly3 – thrust sheet block 2
    # ------------------------------------------------------------------
    xpoly = [990.0,  1825.0, 2165.0, 1170.0 - dx]
    zpoly = [1410.0, 920.0,  1355.0, 1940.0     ]
    vel = afd_vmodel(dx, vel, vhigh, xpoly, zpoly)

    # ------------------------------------------------------------------
    # poly4 – thrust sheet block 3
    # ------------------------------------------------------------------
    xpoly = [1825.0, 2275.0, 2700.0, 2165.0]
    zpoly = [920.0,  390.0,  725.0,  1355.0]
    vel = afd_vmodel(dx, vel, vhigh, xpoly, zpoly)

    # ------------------------------------------------------------------
    # poly5 – thrust sheet block 4 (top / leading tip)
    # ------------------------------------------------------------------
    xpoly = [2275.0, 2500.0, 3106.0, 2700.0]
    zpoly = [390.0,  20.0,   20.0,   725.0 ]
    vel = afd_vmodel(dx, vel, vhigh, xpoly, zpoly)

    # ------------------------------------------------------------------
    # poly6 – sub-thrust target (optional)
    # ------------------------------------------------------------------
    if targetflag:
        xpoly = [
            1400.0, 1479.0, 1590.0, 1730.0, 1925.0, 2071.0,
            2255.0, 2439.0, 2568.0, 2792.0, 2998.0, 3206.0,
            xmax + dx, xmax + dx, 990.0,
        ]
        zpoly = [
            1802.0, 1773.0, 1751.0, 1724.0, 1708.0, 1708.0,
            1708.0, 1724.0, 1729.0, 1740.0, 1743.0, 1751.0,
            1751.0, 1940.0 + dx, 1940.0 + dx,
        ]
        vel = afd_vmodel(dx, vel, (vhigh + vlow) / 2.0, xpoly, zpoly)

    return vel, x, z


# ---------------------------------------------------------------------------
# Quick demo  (mirrors the nargin==0 self-test block in the MATLAB original)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    vel, x, z = thrustmodel(dx=10)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        vel,
        extent=[x[0], x[-1], z[-1], z[0]],
        aspect="auto",
        cmap="seismic",
    )
    fig.colorbar(im, ax=ax, label="velocity (m/s)")
    ax.set_title("Thrust model – colours indicate velocity")
    ax.set_xlabel("distance (m)")
    ax.set_ylabel("depth (m)")
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/thrustmodel_demo.png", dpi=120)
    plt.show()
    print("Done – model shape:", vel.shape)
