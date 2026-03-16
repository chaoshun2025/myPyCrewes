"""ips.py – isotropic phase-shift wavefield extrapolation (stationary).

Mirrors the MATLAB ``ips`` from the Ferguson / Margrave CREWES toolbox.

Reference
---------
Ferguson & Margrave (2005), "Planned seismic imaging using explicit one-way
operators", *Geophysics*, 70(5), S101–S109.
"""

import numpy as np


def ips(phiin, f, dx, parms, dz):
    """
    Isotropic phase-shift extrapolation for a single, stationary velocity.

    Downward-continues the input f-kx wavefield by one depth step *dz*
    using the exact dispersion relation for a homogeneous medium with
    velocity *parms*.

    Parameters
    ----------
    phiin : array_like (complex), shape (nf, nx)
        f-kx spectrum of the input wavefield.  The kx axis must be in
        the wrapped (natural FFT) ordering produced by
        ``np.fft.fftshift(...)``, and only positive frequencies are
        included (one-sided spectrum).
    f : array_like, length nf
        Frequency axis (Hz).
    dx : float
        Spatial sample interval (m).
    parms : float
        Single propagation velocity (m/s).  Must be a scalar; lateral
        velocity variation is handled by the calling function
        :func:`pspi_ips`.
    dz : float
        Depth step (m).  Positive = downward continuation.

    Returns
    -------
    phiout : np.ndarray (complex), shape (nf, nx)
        f-kx spectrum of the extrapolated wavefield.  The output remains
        in the f-kx domain (no spatial IFFT is applied), consistent with
        the MATLAB original.

    Raises
    ------
    ValueError
        If *parms* is not a scalar, or if ``len(f)`` does not match the
        number of rows in *phiin*.

    Notes
    -----
    The vertical wavenumber is computed as

    .. math::

        k_z = \\sqrt{\\left(\\frac{f}{v}\\right)^2 - k_x^2}

    For propagating waves (real :math:`k_z`) this is a pure phase shift.
    For evanescent waves (:math:`k_z^2 < 0`, purely imaginary :math:`k_z`)
    the MATLAB convention

    .. math::

        k_z \\leftarrow \\operatorname{Re}(k_z)
                       + \\operatorname{sgn}(dz)\\,
                         i\\,|\\operatorname{Im}(k_z)|

    is applied, which guarantees exponential decay (not growth) regardless
    of the sign of *dz*.

    The wavenumber axis matches MATLAB's construction::

        kx = fftshift(1 / (2 * nx * dx) * (-nx : 2 : nx-2))
    """
    phiin = np.asarray(phiin, dtype=complex)
    f     = np.asarray(f,     dtype=float).ravel()

    nf, nx = phiin.shape

    parms = np.asarray(parms, dtype=float).ravel()
    if parms.size != 1:
        raise ValueError(
            "ips requires a single (stationary) velocity scalar; "
            "use pspi_ips for laterally varying velocity"
        )
    v = float(parms[0])

    if len(f) != nf:
        raise ValueError(
            f"Frequency axis length ({len(f)}) does not match "
            f"phiin rows ({nf})"
        )

    # ------------------------------------------------------------------
    # Wavenumber axis  (MATLAB: fftshift(1/2/cp/dx * (-cp:2:cp-2)))
    # (-cp:2:cp-2) has exactly cp = nx values: -nx, -nx+2, ..., nx-2
    # ------------------------------------------------------------------
    kx = np.fft.fftshift(
        np.arange(-nx, nx, 2, dtype=float) / (2.0 * nx * dx)
    )   # shape (nx,)

    # ------------------------------------------------------------------
    # Vertical wavenumber  kz = sqrt((f/v)^2 - kx^2)
    # Broadcasting: f[:, None] is (nf,1), kx[None,:] is (1,nx)
    # ------------------------------------------------------------------
    kz2 = (f[:, np.newaxis] / v) ** 2 - kx[np.newaxis, :] ** 2

    # Complex square root: propagating -> real kz, evanescent -> imag kz
    kz = np.sqrt(kz2.astype(complex))

    # MATLAB: kz = real(kz) + sign(dz)*1i*abs(imag(kz))
    # Ensures the evanescent region decays rather than grows, regardless
    # of the sign of dz.
    kz = np.real(kz) + np.sign(dz) * 1j * np.abs(np.imag(kz))

    # ------------------------------------------------------------------
    # Phase-shift operator and application
    # gazx = exp(2*pi*i*dz*kz)
    # phiout = phiin .* gazx   (elementwise, stays in f-kx domain)
    # ------------------------------------------------------------------
    gazx   = np.exp(2j * np.pi * dz * kz)
    phiout = phiin * gazx
    return phiout
