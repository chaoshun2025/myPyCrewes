"""pspi_ips.py – isotropic PSPI one-step wavefield extrapolation.

Mirrors the MATLAB ``pspi_ips`` from the Ferguson / Margrave CREWES toolbox.

Reference
---------
Ferguson & Margrave (2005), "Planned seismic imaging using explicit one-way
operators", *Geophysics*, 70(5), S101–S109.
"""

import numpy as np
from unique_vels import unique_vels
from ips import ips


def pspi_ips(phiin, f, dx, parms, pspi_parms, dz):
    """
    Isotropic PSPI (Phase-Shift Plus Interpolation) wavefield extrapolation.

    Extrapolates the input f-kx wavefield downward by one depth step *dz*
    using the PSPI algorithm:

    1. Identify the unique reference velocities in *pspi_parms*.
    2. For each reference velocity, phase-shift the full wavefield with
       :func:`ips` (which returns the result in the f-kx domain).
    3. Transform the phase-shifted f-kx result to f-x via ``np.fft.fft``
       along the spatial axis (matching MATLAB's ``fft(...,[],2)``).
    4. Accumulate the f-x contributions at the lateral positions assigned
       to each reference velocity.
    5. Apply the thin-lens correction (difference between local velocity
       *parms* and reference velocity *pspi_parms*) in the f-x domain.

    Parameters
    ----------
    phiin : np.ndarray (complex), shape (nf, nx)
        f-kx spectrum of the input wavefield.  The kx axis must use the
        wrapped (natural FFT) ordering and only positive frequencies are
        included.
    f : array_like, length nf
        Frequency axis (Hz).
    dx : float
        Spatial sample interval (m).
    parms : array_like, length nx
        Full (un-smoothed) interval velocity at each lateral position (m/s).
        Used for the thin-lens correction term.
    pspi_parms : array_like, length nx
        Low-pass (blocked / smoothed) velocity at each lateral position (m/s).
        Defines the reference velocities for the phase-shift steps.
    dz : float
        Depth step size (m).

    Returns
    -------
    phiout : np.ndarray (complex), shape (nf, nx)
        f-x spectrum of the extrapolated wavefield.

    Notes
    -----
    The thin-lens correction is applied in the f-x domain:

    .. math::

        \\phi_{\\text{out}} = e^{2\\pi i\\,dz\\,(k - k_0)} \\cdot \\text{temp}

    where :math:`k = f / v_{\\text{parms}}` and
    :math:`k_0 = f / v_{\\text{pspi\\_parms}}` are vertical wavenumbers
    derived from the local and reference velocities respectively, and
    *temp* is the accumulated f-x field from the PSPI summation.

    Domain flow (matching MATLAB exactly)::

        phiin (f-kx)
          └─ ips()          → f-kx   (phase-shifted, proper evanescent decay)
          └─ fft(axis=1)    → f-x    (MATLAB: fft(...,[],2))
          └─ select columns → f-x    (per reference velocity)
          └─ accumulate     → temp   (f-x)
        thin_lens (f-x) * temp (f-x) → phiout (f-x)
    """
    phiin      = np.asarray(phiin,      dtype=complex)
    f          = np.asarray(f,          dtype=float).ravel()
    parms      = np.asarray(parms,      dtype=float).ravel()
    pspi_parms = np.asarray(pspi_parms, dtype=float).ravel()

    nf, nx = phiin.shape

    if len(f) != nf:
        raise ValueError("Frequency axis length does not match phiin rows")

    # ------------------------------------------------------------------
    # Thin-lens vertical wavenumbers (f-x scalars, one per lateral position)
    # k  = f / parms       shape (nf, nx)
    # k0 = f / pspi_parms  shape (nf, nx)
    # ------------------------------------------------------------------
    k  = f[:, np.newaxis] / parms[np.newaxis, :]
    k0 = f[:, np.newaxis] / pspi_parms[np.newaxis, :]

    # ------------------------------------------------------------------
    # PSPI summation over unique reference velocities
    # ------------------------------------------------------------------
    vref_list = unique_vels(pspi_parms)
    temp      = np.zeros((nf, nx), dtype=complex)

    for vref in vref_list:
        inds = np.where(pspi_parms == vref)[0]

        # ips: phase-shift phiin in f-kx domain → result is also f-kx
        phi_fkx = ips(phiin, f, dx, vref, dz)      # shape (nf, nx), f-kx

        # fft along spatial axis: f-kx → f-x  (MATLAB: fft(...,[],2))
        phi_fx  = np.fft.fft(phi_fkx, axis=1)       # shape (nf, nx), f-x

        # Accumulate only the columns assigned to this reference velocity
        temp[:, inds] += phi_fx[:, inds]

    # ------------------------------------------------------------------
    # Thin-lens correction in the f-x domain
    # ------------------------------------------------------------------
    thin_lens = np.exp(2j * np.pi * dz * (k - k0))
    phiout    = thin_lens * temp
    return phiout
