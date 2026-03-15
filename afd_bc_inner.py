"""
afd_bc_inner.py — Absorbing boundary conditions on the inner boundary layer.

Translated from MATLAB afd_bc_inner.m (CREWES / Margrave).

Used with the 9-point Laplacian, which requires two layers of boundary cells.
This function handles the *inner* of those two layers; afd_bc_outer handles
the outermost layer.  For stability: ``v_max * delt / delx < 0.7``.
"""

import numpy as np


def afd_bc_inner(delx:     float,
                 delt:     float,
                 velocity: np.ndarray,
                 snap1:    np.ndarray,
                 snap2:    np.ndarray,
                 snapshot: np.ndarray,
                 boundary: int) -> np.ndarray:
    """
    Apply absorbing BCs to the *inner* boundary layer (row/column index ±1).

    Parameters
    ----------
    delx, delt, velocity, snap1, snap2, snapshot, boundary
        Same meaning as in :func:`afd_bc_outer`.

    Returns
    -------
    snapshot : ndarray, shape (nz, nx)
        Wavefield with inner BCs applied.
    """
    snapshot = snapshot.copy()
    nz, nx   = snap1.shape
    sqrt2    = np.sqrt(2.0)

    def _coef_pos(v):
        return (2.0 * v * delx * delt**2) / (delx + v * delt)

    def _coef_neg(v):
        return (-2.0 * delx * delt**2 * v) / (delx + v * delt)

    # ------------------------------------------------------------------
    # TOP inner row (boundary == 1 only)
    # ------------------------------------------------------------------
    if boundary == 1:
        v = velocity[1, 3:nx - 3]
        snapshot[1, 3:nx - 3] = (
            _coef_pos(v) * (
                snapshot[2, 3:nx - 3] / (2 * delt * delx)
              - snap1[2, 3:nx - 3]   / (2 * delx * delt)
              + snap1[1, 3:nx - 3]   / (2 * delt * delx)
            )
            + _coef_pos(v) * (
                -1.0 / (2.0 * delt**2 * v) * (
                    -2 * snap2[1, 3:nx - 3]
                    + snap1[1, 3:nx - 3]
                    - 2 * snap2[2, 3:nx - 3]
                    + snap1[2, 3:nx - 3]
                    + snapshot[2, 3:nx - 3]
                )
            )
            + _coef_pos(v) * (
                v / (4 * delx**2) * (
                    snapshot[2, 4:nx - 2]
                  + snap1[1, 4:nx - 2]
                  + snapshot[2, 2:nx - 4]
                  - 2 * snapshot[2, 3:nx - 3]
                  - 2 * snap1[1, 3:nx - 3]
                  + snap1[1, 2:nx - 4]
                )
            )
        )

    # ------------------------------------------------------------------
    # BOTTOM inner row
    # ------------------------------------------------------------------
    v = velocity[nz - 2, 3:nx - 3]
    snapshot[nz - 2, 3:nx - 3] = (
        _coef_neg(v) * (
            -snapshot[nz - 3, 3:nx - 3] / (2 * delt * delx)
            - snap1[nz - 2, 3:nx - 3]  / (2 * delt * delx)
            + snap1[nz - 3, 3:nx - 3]  / (2 * delt * delx)
        )
        + _coef_neg(v) * (
            1.0 / (2 * delt**2 * v) * (
                -2 * snap2[nz - 2, 3:nx - 3]
                + snap1[nz - 2, 3:nx - 3]
                + snapshot[nz - 3, 3:nx - 3]
                - 2 * snap2[nz - 3, 3:nx - 3]
                + snap1[nz - 3, 3:nx - 3]
            )
        )
        + _coef_neg(v) * (
            -v / (4 * delx**2) * (
                snapshot[nz - 3, 4:nx - 2]
              - 2 * snapshot[nz - 3, 3:nx - 3]
              + snapshot[nz - 3, 2:nx - 4]
              + snap1[nz - 2, 4:nx - 2]
              - 2 * snap1[nz - 2, 3:nx - 3]
              + snap1[nz - 2, 2:nx - 4]
            )
        )
    )

    # ------------------------------------------------------------------
    # RIGHT inner column
    # ------------------------------------------------------------------
    v = velocity[3:nz - 3, nx - 2]
    snapshot[3:nz - 3, nx - 2] = (
        _coef_neg(v) * (
            -snapshot[3:nz - 3, nx - 3] / (2 * delt * delx)
            - snap1[3:nz - 3, nx - 2]  / (2 * delt * delx)
            + snap1[3:nz - 3, nx - 3]  / (2 * delt * delx)
        )
        + _coef_neg(v) * (
            1.0 / (2 * delt**2 * v) * (
                -2 * snap2[3:nz - 3, nx - 2]
                + snap1[3:nz - 3, nx - 2]
                + snapshot[3:nz - 3, nx - 3]
                - 2 * snap2[3:nz - 3, nx - 3]
                + snap1[3:nz - 3, nx - 3]
            )
        )
        + _coef_neg(v) * (
            -v / (4 * delx**2) * (
                snapshot[4:nz - 2, nx - 3]
              - 2 * snapshot[3:nz - 3, nx - 3]
              + snapshot[2:nz - 4, nx - 3]
              + snap1[4:nz - 2, nx - 2]
              - 2 * snap1[3:nz - 3, nx - 2]
              + snap1[2:nz - 4, nx - 2]
            )
        )
    )

    # ------------------------------------------------------------------
    # LEFT inner column
    # ------------------------------------------------------------------
    v = velocity[3:nz - 3, 1]
    snapshot[3:nz - 3, 1] = (
        _coef_pos(v) * (
            snapshot[3:nz - 3, 2] / (2 * delt * delx)
          - snap1[3:nz - 3, 2]   / (2 * delt * delx)
          + snap1[3:nz - 3, 1]   / (2 * delt * delx)
        )
        + _coef_pos(v) * (
            -1.0 / (2 * delt**2 * v) * (
                -2 * snap2[3:nz - 3, 1]
                + snap1[3:nz - 3, 1]
                + snapshot[3:nz - 3, 2]
                - 2 * snap2[3:nz - 3, 2]
                + snap1[3:nz - 3, 2]
            )
        )
        + _coef_pos(v) * (
            v / (4 * delx**2) * (
                snapshot[4:nz - 2, 2]
              - 2 * snapshot[3:nz - 3, 2]
              + snapshot[2:nz - 4, 2]
              + snap1[4:nz - 2, 1]
              - 2 * snap1[3:nz - 3, 1]
              + snap1[2:nz - 4, 1]
            )
        )
    )

    # ------------------------------------------------------------------
    # CORNER cells — lower-right inner
    # ------------------------------------------------------------------
    def _corner(v, above, left, prev):
        return v * delt * delx / (2.0 * v * delt + sqrt2 * delx) * (
            above / delx + left / delx + sqrt2 / (v * delt) * prev
        )

    snapshot[nz - 3, nx - 2] = _corner(
        velocity[nz - 3, nx - 2],
        snapshot[nz - 4, nx - 2], snapshot[nz - 3, nx - 3],
        snap2[nz - 3, nx - 2])

    snapshot[nz - 2, nx - 3] = _corner(
        velocity[nz - 2, nx - 3],
        snapshot[nz - 3, nx - 3], snapshot[nz - 2, nx - 4],
        snap2[nz - 2, nx - 3])

    snapshot[nz - 2, nx - 2] = _corner(
        velocity[nz - 2, nx - 2],
        snapshot[nz - 3, nx - 2], snapshot[nz - 2, nx - 3],
        snap2[nz - 2, nx - 2])

    # lower-left inner
    snapshot[nz - 3, 1] = _corner(
        velocity[nz - 3, 1],
        snapshot[nz - 4, 1], snapshot[nz - 3, 2],
        snap2[nz - 3, 1])

    snapshot[nz - 2, 2] = _corner(
        velocity[nz - 2, 2],
        snapshot[nz - 3, 2], snapshot[nz - 2, 3],
        snap2[nz - 2, 2])

    snapshot[nz - 2, 1] = _corner(
        velocity[nz - 2, 1],
        snapshot[nz - 3, 1], snapshot[nz - 2, 2],
        snap2[nz - 2, 1])

    if boundary == 1:
        # upper-right inner
        snapshot[2, nx - 2] = _corner(
            velocity[2, nx - 2],
            snapshot[3, nx - 2], snapshot[2, nx - 3],
            snap2[2, nx - 2])

        snapshot[1, nx - 3] = _corner(
            velocity[1, nx - 3],
            snapshot[2, nx - 3], snapshot[1, nx - 4],
            snap2[1, nx - 3])

        snapshot[1, nx - 2] = _corner(
            velocity[1, nx - 2],
            snapshot[2, nx - 2], snapshot[1, nx - 3],
            snap2[1, nx - 2])

        # upper-left inner
        snapshot[2, 1] = _corner(
            velocity[2, 1],
            snapshot[3, 1], snapshot[2, 2],
            snap2[2, 1])

        snapshot[1, 2] = _corner(
            velocity[1, 2],
            snapshot[2, 2], snapshot[1, 3],
            snap2[1, 2])

        snapshot[1, 1] = _corner(
            velocity[1, 1],
            snapshot[2, 1], snapshot[1, 2],
            snap2[1, 1])

    return snapshot
