"""ss_mig.py – pre-stack split-step depth migration.

Mirrors the MATLAB ``ss_mig`` from the Ferguson / Margrave CREWES toolbox.

Reference
---------
Ferguson & Margrave (2005), "Planned seismic imaging using explicit one-way
operators", *Geophysics*, 70(5), S101–S109.
"""

import numpy as np
from ss_ips import ss_ips


def ss_mig(fdata, f, parms, dx, dz):
    """
    Pre-stack split-step depth migration.

    Migrates a band-limited, one-sided f-kx data matrix using the
    split-step phase-shift algorithm.  Both the recorded wavefield and a
    modelled point source are downward-continued simultaneously using
    :func:`ss_ips`.

    Parameters
    ----------
    fdata : array_like (complex), shape (nf, nx)
        f-kx spectrum of the zero-offset data.  The frequency axis must
        be band-limited and positive-only; the kx axis must be **wrapped**
        (natural FFT ordering).
    f : array_like, length nf
        Frequency axis (Hz) corresponding to the rows of *fdata*.
    parms : array_like, shape (Nz, nx)
        Interval velocity model (m/s).  Each row is a constant-depth
        slice.
    dx : float
        Trace spacing (m).
    dz : float
        Depth step (m).

    Returns
    -------
    mdata : np.ndarray (float), shape (Nz, nx)
        Depth-migrated image.  Row 0 is the surface; row ``j+1`` holds
        the image at depth ``(j+1) * dz``.

    Notes
    -----
    **Source model**: a unit point source is placed at the lateral midpoint
    ``round(nx / 2)`` (0-based index), matching MATLAB's
    ``round(cd/2)+1`` (1-based).

    **Domain flow** per depth step (:func:`ss_ips` returns f-kx):

    1. ``ftemp = ss_ips(fdata, ..., +dz)``  – receiver wavefield (f-kx).
    2. ``fdata_fx = fft(ftemp, axis=1)``    – convert to f-x for imaging.
    3. ``stemp = ss_ips(fsou,  ..., −dz)``  – source wavefield (f-kx).
    4. ``fsou_fx  = fft(stemp, axis=1)``    – convert to f-x.
    5. Imaging: ``rtemp = fdata_fx * conj(fsou_fx)`` (f-x cross-correlation).
    6. Accumulate image row:
       ``real(sum(rtemp) + sum(rtemp[:nf-1,:])) / (2*nf-1) / (2*pi)``.
    7. ``fdata = ftemp``, ``fsou = stemp``  – restore f-kx for next step.

    The key difference from :func:`pspi_mig` is that ``ss_ips`` returns
    f-kx (not f-x), so an explicit ``fft`` is needed before applying the
    imaging condition.
    """
    fdata  = np.asarray(fdata, dtype=complex)
    f      = np.asarray(f,     dtype=float).ravel()
    parms  = np.asarray(parms, dtype=float)

    Nz, Nx = parms.shape
    rd, cd  = fdata.shape

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
    # MATLAB: temp(:, round(cd/2)+1)=1 (1-based) -> Python: round(cd/2)
    # ------------------------------------------------------------------
    temp = np.zeros((rd, cd), dtype=complex)
    temp[:, round(cd / 2)] = 1.0
    fsou = np.fft.ifft(temp, axis=1)   # f-kx

    mdata = np.zeros((Nz, cd))

    for j in range(Nz - 1):
        print(f" ss prestack mig working on depth {j + 1} of {Nz}")

        # ss_ips returns f-kx
        ftemp = ss_ips(fdata, f, dx, parms[j, :],  dz)
        stemp = ss_ips(fsou,  f, dx, parms[j, :], -dz)

        # fft(f-kx) along spatial axis -> f-x for imaging condition
        # MATLAB: fdata = fft(ftemp,[],2); fsou = fft(stemp,[],2)
        fdata_fx = np.fft.fft(ftemp, axis=1)
        fsou_fx  = np.fft.fft(stemp, axis=1)

        # Imaging condition: cross-correlation in f-x domain
        rtemp = fdata_fx * np.conj(fsou_fx)

        # Accumulate over frequency axis
        # MATLAB: real(sum(rtemp)+sum(rtemp(1:rd-1,:))) / (2*rd-1) / 2/pi
        mdata[j + 1, :] = (
            np.real(rtemp.sum(axis=0) + rtemp[:rd - 1, :].sum(axis=0))
            / (2 * rd - 1)
            / (2.0 * np.pi)
        )

        # Restore f-kx for next depth step (ss_ips inputs must be f-kx)
        fdata = ftemp
        fsou  = stemp

    return mdata
