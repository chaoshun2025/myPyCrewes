"""
afd_reflect.py — Compute normal-incidence reflectivity from a velocity model.

Translated from MATLAB afd_reflect.m (CREWES / Margrave).
"""

import numpy as np


def afd_reflect(velocity: np.ndarray, clipn: int) -> np.ndarray:
    """
    Compute the normal-incidence reflectivity of a velocity model.

    The reflectivity is derived from the spatial gradient of ln(velocity),
    which is proportional to the impedance contrast for a constant-density
    medium:

    .. math::

        r = 0.5 \\cdot \\text{sign}(r_z) \\cdot
            \\sqrt{r_x^2 + r_z^2}, \\quad
            r_x, r_z = \\nabla \\ln(v)

    Parameters
    ----------
    velocity : ndarray, shape (nz, nx)
        Velocity model (m/s or consistent units).
    clipn    : int
        Number of border layers to zero out to suppress artefacts near
        the absorbing boundaries.  Typical value: 5.  Use 0 to skip.

    Returns
    -------
    r : ndarray, shape (nz, nx)
        Reflectivity, with *clipn* border layers set to zero.
    """
    velocity = np.asarray(velocity, dtype=float)
    nz, nx   = velocity.shape

    # gradient of ln(v) — numpy.gradient returns [dz, dx]
    log_v         = np.log(velocity)
    rz, rx        = np.gradient(log_v)      # dz first (row axis), dx second

    tmp = 0.5 * np.sign(rz) * np.sqrt(rx**2 + rz**2)

    r = np.zeros_like(tmp)
    r[clipn: nz - clipn, clipn: nx - clipn] = \
        tmp[clipn: nz - clipn, clipn: nx - clipn]

    return r
