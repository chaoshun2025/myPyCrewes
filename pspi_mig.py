"""pspi_mig.py – pre-stack PSPI depth migration.

Mirrors the MATLAB ``pspi_mig`` from the Ferguson / Margrave CREWES toolbox.

Reference
---------
Ferguson & Margrave (2005), "Planned seismic imaging using explicit one-way
operators", *Geophysics*, 70(5), S101–S109.
"""

import numpy as np
from pspi_ips import pspi_ips


def pspi_mig(fdata, f, parms, pspi_parms, dx, dz):
    """
    Pre-stack PSPI depth migration.

    Migrates a band-limited, one-sided f-kx data matrix using the
    Phase-Shift Plus Interpolation (PSPI) algorithm.  Both the recorded
    wavefield (*fdata*) and a modelled point source are downward-continued
    simultaneously.  An imaging condition is applied at every depth step.

    Parameters
    ----------
    fdata : array_like (complex), shape (nf, nx)
        f-kx spectrum of the zero-offset data.  The frequency axis must
        be band-limited and positive-only; the kx axis must be **wrapped**
        (natural FFT ordering, not fftshifted).
    f : array_like, length nf
        Frequency axis (Hz) corresponding to the rows of *fdata*.
    parms : array_like, shape (Nz, nx)
        Full (un-smoothed) interval velocity model (m/s).  Each row is
        a constant-depth slice.
    pspi_parms : array_like, shape (Nz, nx)
        Blocked (laterally smoothed) velocity model (m/s), same shape as
        *parms*.  Used by :func:`pspi_ips` to select reference velocities.
    dx : float
        Trace spacing (m).
    dz : float
        Depth step (m).

    Returns
    -------
    mdata : np.ndarray (float), shape (Nz, nx)
        Depth-migrated image.  Row 0 is surface (zero depth); row ``j+1``
        holds the image at depth ``(j+1) * dz``.

    Notes
    -----
    **Source model**: a unit point source is placed at the lateral midpoint
    ``round(nx / 2)`` (0-based index), matching MATLAB's
    ``round(cd/2)+1`` (1-based).

    **Domain flow** per depth step:

    1. ``ftemp = pspi_ips(fdata, ..., +dz)`` – receiver wavefield (f-x).
    2. ``stemp = pspi_ips(fsou,  ..., −dz)`` – source wavefield upward
       (f-x).
    3. Imaging: ``rtemp = ftemp * conj(stemp)`` (f-x cross-correlation).
    4. Accumulate image row:
       ``real(sum(rtemp) + sum(rtemp[:nf-1,:])) / (2*nf-1) / (2*pi)``.
    5. ``fdata = ifft(ftemp, axis=1)`` – restore f-kx for next step.
    6. ``fsou  = ifft(stemp, axis=1)`` – same for source.
    """
    fdata      = np.asarray(fdata,      dtype=complex)
    f          = np.asarray(f,          dtype=float).ravel()
    parms      = np.asarray(parms,      dtype=float)
    pspi_parms = np.asarray(pspi_parms, dtype=float)

    Nz, Nx = parms.shape
    rd, cd = fdata.shape

    if len(f) != rd:
        raise ValueError(
            f"Frequency axis length ({len(f)}) does not match "
            f"fdata rows ({rd})"
        )
    if Nx != cd:
        raise ValueError(
            f"Velocity model columns ({Nx}) do not match fdata columns ({cd})"
        )

    # ------------------------------------------------------------------
    # Build point source: unit impulse at lateral midpoint, then ifft
    # MATLAB: temp(:, round(cd/2)+1) = 1  (1-based)
    #      -> Python index: round(cd/2)  (0-based)
    # ------------------------------------------------------------------
    temp = np.zeros((rd, cd), dtype=complex)
    temp[:, round(cd / 2)] = 1.0
    fsou = np.fft.ifft(temp, axis=1)   # f-kx

    mdata = np.zeros((Nz, cd))

    for j in range(Nz - 1):
        print(f" pspi prestack mig working on depth {j + 1} of {Nz}")

        # pspi_ips returns f-x
        ftemp = pspi_ips(fdata, f, dx, parms[j, :],      pspi_parms[j, :],  dz)
        stemp = pspi_ips(fsou,  f, dx, parms[j, :],      pspi_parms[j, :], -dz)

        # Imaging condition (cross-correlation, f-x domain)
        rtemp = ftemp * np.conj(stemp)

        # Accumulate over frequency axis: sum all rows + sum first rd-1 rows
        # MATLAB: real(sum(rtemp)+sum(rtemp(1:rd-1,:))) / (2*rd-1) / 2/pi
        mdata[j + 1, :] = (
            np.real(rtemp.sum(axis=0) + rtemp[:rd - 1, :].sum(axis=0))
            / (2 * rd - 1)
            / (2.0 * np.pi)
        )

        # ifft(f-x) along spatial axis restores f-kx for next depth step
        fdata = np.fft.ifft(ftemp, axis=1)
        fsou  = np.fft.ifft(stemp, axis=1)

    return mdata
