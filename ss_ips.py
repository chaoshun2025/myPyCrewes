"""ss_ips.py – isotropic split-step wavefield extrapolation (stationary reference).

Mirrors the MATLAB ``ss_ips`` from the Ferguson / Margrave CREWES toolbox.

Reference
---------
Ferguson & Margrave (2005), "Planned seismic imaging using explicit one-way
operators", *Geophysics*, 70(5), S101–S109.
"""

import numpy as np


def ss_ips(phiin, f, dx, parms, dz):
    """
    Isotropic split-step phase-shift extrapolation.

    Downward-continues the input f-kx wavefield by one depth step *dz*
    using the split-step algorithm:

    1. Compute a **reference phase shift** in the f-kx domain using a
       single reference velocity ``v_ref = 0.99 * min(parms)`` (the
       focusing operator).
    2. Compute a **thin-lens correction** in the f-x domain using the
       local velocity *parms* (lateral variation correction).
    3. Apply both corrections and return the result in the **f-kx domain**.

    Parameters
    ----------
    phiin : array_like (complex), shape (nf, nx)
        f-kx spectrum of the input wavefield.  The kx axis must be in
        the wrapped (natural FFT) ordering, and only positive frequencies
        are included.
    f : array_like, length nf
        Frequency axis (Hz).
    dx : float
        Spatial sample interval (m).
    parms : array_like, length nx
        Lateral interval velocity profile at the current depth (m/s).
        The minimum of *parms* defines the reference velocity.
    dz : float
        Depth step (m).  Positive = downward continuation.

    Returns
    -------
    phiout : np.ndarray (complex), shape (nf, nx)
        f-kx spectrum of the extrapolated wavefield.

    Notes
    -----
    The reference velocity is ``v_ref = 0.99 * min(parms)``, chosen to
    lie slightly below the minimum model velocity so that all propagating
    waves receive a non-trivial focusing correction.

    The split-step operator is::

        W0       = thin_lens * fft(phiin, axis=1)   # f-x domain
        w0       = ifft(W0, axis=1)                 # back to f-kx
        phiout   = focus * w0                       # focusing phase shift

    where::

        thin_lens = exp(2π i dz (k − k0))
        focus     = exp(2π i dz kz0)
        k         = f / parms   (shape nf × nx, lateral variation)
        k0        = f / v_ref   (shape nf × nx, constant across x)
        kz0       = sqrt(k0² − kx²) with evanescent → decay convention

    Domain flow: f-kx → (thin-lens via fft/ifft) → f-kx → (focus) → f-kx.
    """
    phiin  = np.asarray(phiin, dtype=complex)
    f      = np.asarray(f,     dtype=float).ravel()
    parms  = np.asarray(parms, dtype=float).ravel()

    nf, nx = phiin.shape

    if len(f) != nf:
        raise ValueError(
            f"Frequency axis length ({len(f)}) does not match "
            f"phiin rows ({nf})"
        )

    # ------------------------------------------------------------------
    # Wavenumber axis  (MATLAB: fftshift(1/2/cp/dx * (-cp:2:cp-2)))
    # ------------------------------------------------------------------
    kx = np.fft.fftshift(
        np.arange(-nx, nx, 2, dtype=float) / (2.0 * nx * dx)
    )   # shape (nx,)

    # ------------------------------------------------------------------
    # Local and reference vertical wavenumbers
    # k  = f / parms       shape (nf, nx) – lateral variation
    # k0 = f / v_ref       shape (nf, nx) – constant reference
    # ------------------------------------------------------------------
    v_ref = 0.99 * float(np.min(parms))
    k  = f[:, np.newaxis] * (1.0 / parms[np.newaxis, :])    # (nf, nx)
    k0 = f[:, np.newaxis] * (1.0 / v_ref) * np.ones((1, nx))  # (nf, nx)

    # ------------------------------------------------------------------
    # Reference vertical wavenumber  kz0 = sqrt(k0^2 - kx^2)
    # ------------------------------------------------------------------
    kz0 = np.sqrt((k0 ** 2 - kx[np.newaxis, :] ** 2).astype(complex))

    # MATLAB evanescent convention: ensure decay not growth
    kz0 = np.real(kz0) + np.sign(dz) * 1j * np.abs(np.imag(kz0))

    # ------------------------------------------------------------------
    # Focusing operator (f-kx domain) and thin-lens correction (f-x)
    # ------------------------------------------------------------------
    focus      = np.exp(2j * np.pi * dz * kz0)           # (nf, nx) f-kx
    thin_lens  = np.exp(2j * np.pi * dz * (k - k0))      # (nf, nx) f-x

    # ------------------------------------------------------------------
    # Split-step extrapolation
    #   W0     = thin_lens * fft(phiin, axis=1)   f-x
    #   w0     = ifft(W0,  axis=1)                f-kx
    #   phiout = focus * w0                       f-kx
    # ------------------------------------------------------------------
    W0     = thin_lens * np.fft.fft(phiin,  axis=1)
    w0     = np.fft.ifft(W0, axis=1)
    phiout = focus * w0
    return phiout
