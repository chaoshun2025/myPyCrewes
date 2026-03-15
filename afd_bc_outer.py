"""
afd_bc_outer.py — Absorbing boundary conditions on the outer grid boundary.

Translated from MATLAB afd_bc_outer.m (CREWES / Margrave).

These BCs are used with the 5-point Laplacian (one layer of boundary cells).
For stability: ``v_max * delt / delx < 0.7``.
"""

import numpy as np


def afd_bc_outer(delx:     float,
                 delt:     float,
                 velocity: np.ndarray,
                 snap1:    np.ndarray,
                 snap2:    np.ndarray,
                 snapshot: np.ndarray,
                 boundary: int) -> np.ndarray:
    """
    Apply absorbing BCs to the *outer* layer of rows and columns.

    Parameters
    ----------
    delx     : float
        Grid spacing (m).
    delt     : float
        Time step (s).
    velocity : ndarray, shape (nz, nx)
        Velocity model (m/s).  Already halved for exploding-reflector.
    snap1    : ndarray, shape (nz, nx)
        Wavefield at t − 2·delt.
    snap2    : ndarray, shape (nz, nx)
        Wavefield at t − delt.
    snapshot : ndarray, shape (nz, nx)
        Wavefield at t (interior already propagated, boundaries to be filled).
    boundary : int
        ``1`` — all four sides absorbing.
        ``2`` — three sides absorbing, top free (for surface sources).

    Returns
    -------
    snapshot : ndarray, shape (nz, nx)
        Wavefield with absorbing boundary conditions applied.
    """
    snapshot = snapshot.copy()
    nz, nx   = snap1.shape
    sqrt2    = np.sqrt(2.0)

    # ------------------------------------------------------------------
    # Interior edge helpers (re-used in multiple boundary segments)
    # ------------------------------------------------------------------
    def _coef_pos(v, ix, iz):
        """Coefficient for boundary pointing *into* the domain (Clayton-Engquist)."""
        return (2.0 * v * delx * delt**2) / (delx + v * delt)

    def _coef_neg(v, ix, iz):
        return (-2.0 * delx * delt**2 * v) / (delx + v * delt)

    # ------------------------------------------------------------------
    # TOP boundary  (boundary == 1 only)
    # ------------------------------------------------------------------
    if boundary == 1:
        v = velocity[0, 2:nx - 2]
        snapshot[0, 2:nx - 2] = (
            _coef_pos(v, None, 0) * (
                snapshot[1, 2:nx - 2] / (2 * delt * delx)
              - snap1[1, 2:nx - 2]   / (2 * delx * delt)
              + snap1[0, 2:nx - 2]   / (2 * delt * delx)
            )
            + _coef_pos(v, None, 0) * (
                -1.0 / (2.0 * delt**2 * v) * (
                    -2 * snap2[0, 2:nx - 2]
                    + snap1[0, 2:nx - 2]
                    - 2 * snap2[1, 2:nx - 2]
                    + snap1[1, 2:nx - 2]
                    + snapshot[1, 2:nx - 2]
                )
            )
            + _coef_pos(v, None, 0) * (
                v / (4 * delx**2) * (
                    snapshot[1, 3:nx - 1]
                  + snap1[0, 3:nx - 1]
                  + snapshot[1, 1:nx - 3]
                  - 2 * snapshot[1, 2:nx - 2]
                  - 2 * snap1[0, 2:nx - 2]
                  + snap1[0, 1:nx - 3]
                )
            )
        )

    # ------------------------------------------------------------------
    # BOTTOM boundary
    # ------------------------------------------------------------------
    v = velocity[nz - 1, 2:nx - 2]
    snapshot[nz - 1, 2:nx - 2] = (
        _coef_neg(v, None, nz - 1) * (
            -snapshot[nz - 2, 2:nx - 2] / (2 * delt * delx)
            - snap1[nz - 1, 2:nx - 2]  / (2 * delt * delx)
            + snap1[nz - 2, 2:nx - 2]  / (2 * delt * delx)
        )
        + _coef_neg(v, None, nz - 1) * (
            1.0 / (2 * delt**2 * v) * (
                -2 * snap2[nz - 1, 2:nx - 2]
                + snap1[nz - 1, 2:nx - 2]
                + snapshot[nz - 2, 2:nx - 2]
                - 2 * snap2[nz - 2, 2:nx - 2]
                + snap1[nz - 2, 2:nx - 2]
            )
        )
        + _coef_neg(v, None, nz - 1) * (
            -v / (4 * delx**2) * (
                snapshot[nz - 2, 3:nx - 1]
              - 2 * snapshot[nz - 2, 2:nx - 2]
              + snapshot[nz - 2, 1:nx - 3]
              + snap1[nz - 1, 3:nx - 1]
              - 2 * snap1[nz - 1, 2:nx - 2]
              + snap1[nz - 1, 1:nx - 3]
            )
        )
    )

    # ------------------------------------------------------------------
    # RIGHT boundary
    # ------------------------------------------------------------------
    v = velocity[2:nz - 2, nx - 1]
    snapshot[2:nz - 2, nx - 1] = (
        _coef_neg(v, nx - 1, None) * (
            -snapshot[2:nz - 2, nx - 2] / (2 * delt * delx)
            - snap1[2:nz - 2, nx - 1]  / (2 * delt * delx)
            + snap1[2:nz - 2, nx - 2]  / (2 * delt * delx)
        )
        + _coef_neg(v, nx - 1, None) * (
            1.0 / (2 * delt**2 * v) * (
                -2 * snap2[2:nz - 2, nx - 1]
                + snap1[2:nz - 2, nx - 1]
                + snapshot[2:nz - 2, nx - 2]
                - 2 * snap2[2:nz - 2, nx - 2]
                + snap1[2:nz - 2, nx - 2]
            )
        )
        + _coef_neg(v, nx - 1, None) * (
            -v / (4 * delx**2) * (
                snapshot[3:nz - 1, nx - 2]
              - 2 * snapshot[2:nz - 2, nx - 2]
              + snapshot[1:nz - 3, nx - 2]
              + snap1[3:nz - 1, nx - 1]
              - 2 * snap1[2:nz - 2, nx - 1]
              + snap1[1:nz - 3, nx - 1]
            )
        )
    )

    # ------------------------------------------------------------------
    # LEFT boundary
    # ------------------------------------------------------------------
    v = velocity[2:nz - 2, 0]
    snapshot[2:nz - 2, 0] = (
        _coef_pos(v, 0, None) * (
            snapshot[2:nz - 2, 1] / (2 * delt * delx)
          - snap1[2:nz - 2, 1]   / (2 * delt * delx)
          + snap1[2:nz - 2, 0]   / (2 * delt * delx)
        )
        + _coef_pos(v, 0, None) * (
            -1.0 / (2 * delt**2 * v) * (
                -2 * snap2[2:nz - 2, 0]
                + snap1[2:nz - 2, 0]
                + snapshot[2:nz - 2, 1]
                - 2 * snap2[2:nz - 2, 1]
                + snap1[2:nz - 2, 1]
            )
        )
        + _coef_pos(v, 0, None) * (
            v / (4 * delx**2) * (
                snapshot[3:nz - 1, 1]
              - 2 * snapshot[2:nz - 2, 1]
              + snapshot[1:nz - 3, 1]
              + snap1[3:nz - 1, 0]
              - 2 * snap1[2:nz - 2, 0]
              + snap1[1:nz - 3, 0]
            )
        )
    )

    # ------------------------------------------------------------------
    # CORNER cells — lower-right
    # ------------------------------------------------------------------
    def _corner(v, above, left, prev):
        return v * delt * delx / (2.0 * v * delt + sqrt2 * delx) * (
            above / delx + left / delx + sqrt2 / (v * delt) * prev
        )

    snapshot[nz - 2, nx - 1] = _corner(
        velocity[nz - 2, nx - 1],
        snapshot[nz - 3, nx - 1], snapshot[nz - 2, nx - 2],
        snap2[nz - 2, nx - 1])

    snapshot[nz - 1, nx - 2] = _corner(
        velocity[nz - 1, nx - 2],
        snapshot[nz - 2, nx - 2], snapshot[nz - 1, nx - 3],
        snap2[nz - 1, nx - 2])

    snapshot[nz - 1, nx - 1] = _corner(
        velocity[nz - 1, nx - 1],
        snapshot[nz - 2, nx - 1], snapshot[nz - 1, nx - 2],
        snap2[nz - 1, nx - 1])

    # lower-left
    snapshot[nz - 2, 0] = _corner(
        velocity[nz - 2, 0],
        snapshot[nz - 3, 0], snapshot[nz - 2, 1],
        snap2[nz - 2, 0])

    snapshot[nz - 1, 1] = _corner(
        velocity[nz - 1, 1],
        snapshot[nz - 2, 1], snapshot[nz - 1, 2],
        snap2[nz - 1, 1])

    snapshot[nz - 1, 0] = _corner(
        velocity[nz - 1, 0],
        snapshot[nz - 2, 0], snapshot[nz - 1, 1],
        snap2[nz - 1, 0])

    if boundary == 1:
        # upper-right
        snapshot[1, nx - 1] = _corner(
            velocity[1, nx - 1],
            snapshot[2, nx - 1], snapshot[1, nx - 2],
            snap2[1, nx - 1])

        snapshot[0, nx - 2] = _corner(
            velocity[0, nx - 2],
            snapshot[1, nx - 2], snapshot[0, nx - 3],
            snap2[0, nx - 2])

        snapshot[0, nx - 1] = _corner(
            velocity[0, nx - 1],
            snapshot[1, nx - 1], snapshot[0, nx - 2],
            snap2[0, nx - 1])

        # upper-left
        snapshot[1, 0] = _corner(
            velocity[1, 0],
            snapshot[2, 0], snapshot[1, 1],
            snap2[1, 0])

        snapshot[0, 1] = _corner(
            velocity[0, 1],
            snapshot[1, 1], snapshot[0, 2],
            snap2[0, 1])

        snapshot[0, 0] = _corner(
            velocity[0, 0],
            snapshot[1, 0], snapshot[0, 1],
            snap2[0, 0])

    return snapshot
