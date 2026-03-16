"""splitstepf_mig.py – split-step Fourier depth migration for v(z) media.

Mirrors the MATLAB ``splitstepf_mig`` from the Margrave CREWES toolbox.
"""

import time
import numpy as np
from fktran import fktran
from near import near
from pwlint import pwlint


def splitstepf_mig(seis, t, x, velmod, zv, dz, zmax, fmax):
    """
    Split-step Fourier depth migration through a v(z) medium.

    Performs depth migration using a phase-shift approach that handles
    mild lateral velocity variation via a split-step correction.  For
    each depth step the algorithm applies:

    1. A **focusing phase shift** in the f-kx domain using the mean
       velocity at that depth.
    2. A **static phase shift** (residual correction) in the f-x domain
       using the local lateral velocity.

    Parameters
    ----------
    seis : array_like, shape (nt, nx)
        Input seismic data matrix.  One trace per column.
    t : array_like, length nt
        Time coordinate vector (s); must be regularly sampled.
    x : array_like, length nx
        Lateral coordinate vector (m); must be regularly sampled.
    velmod : array_like, shape (nzv, nx)
        Velocity model (m/s).  Must have the same number of columns as
        *seis* and its depth axis must span 0 to *zmax*.
    zv : array_like, length nzv
        Depth coordinate vector (m) for *velmod*.
    dz : float
        Depth step size (m).
    zmax : float
        Maximum depth to migrate to (m).
    fmax : float
        Maximum frequency to migrate (Hz).

    Returns
    -------
    seismig : np.ndarray (float), shape (nz, nx)
        Depth-migrated image.
    zmig : np.ndarray
        Depth coordinate vector (m), length nz = ``len(0:dz:zmax)``.

    Notes
    -----
    **Exploding-reflector convention**: the velocity model is halved
    (``v = 0.5 * velmod[izv, :]``) at each depth step.

    **f-kx transform**: uses :func:`fktran` with ``ishift=0`` so the kx
    axis is **wrapped** (natural FFT ordering).  ``ntpad`` and ``nxpad``
    are the next powers of 2 above ``len(t)`` and ``len(x)`` respectively.

    **Inner loops** match the MATLAB original exactly:

    *Loop over kx columns* – focusing phase shift (evanescent zeroed):

    .. code-block:: text

        fev  = |kx[jk]| * vm          (first non-evanescent frequency)
        nfev = max(round(fev/df), 0)  (0-based index)
        psf  = (2π dz f[nfev:] / vm) * (sqrt(1 − vm²kx²/f²) − 1)
        phi[nfev:, jk] *= exp(i * psf)
        phi[:nfev, jk]  = 0

    *Loop over frequency rows* – static shift and imaging:

    .. code-block:: text

        tmp = fft(phi[jf, :nx])
        tmp *= exp(i * 2π * f[jf] * dz / v)
        seismig[iz, :] += 2 * real(tmp)
        phi[jf, :] = [ifft(tmp), zeros(nkxpad - nx)]

    The static-shift frequency index ``f[jf]`` (0-based) matches MATLAB's
    ``f(jf)`` (1-based), preserving the original off-by-one convention
    where the first row (DC) receives a zero shift.
    """
    seis   = np.asarray(seis,   dtype=float)
    t      = np.asarray(t,      dtype=float).ravel()
    x      = np.asarray(x,      dtype=float).ravel()
    velmod = np.asarray(velmod, dtype=float)
    zv     = np.asarray(zv,     dtype=float).ravel()

    nsamp, ntr = seis.shape
    nx = len(x)

    # Depth axis
    z   = np.arange(0.0, zmax + dz * 0.5, dz)
    nz  = len(z)

    # ------------------------------------------------------------------
    # Forward f-k transform  (ishift=0 → wrapped kx)
    # MATLAB: fktran(seis,t,x, 2^nextpow2(t), 2^nextpow2(x), 0, 0)
    # ------------------------------------------------------------------
    ntpad = int(2 ** np.ceil(np.log2(len(t))))
    nxpad = int(2 ** np.ceil(np.log2(len(x))))
    print("fk transform")
    phi, f, kx = fktran(seis, t, x, ntpad, nxpad, 0.0, 0)
    # phi shape: (nf, nkxpad), kx is wrapped

    kx2 = kx ** 2

    df      = f[1] - f[0]
    # nfmax: index of last frequency <= fmax (1-based in MATLAB -> 0-based +1 here)
    # MATLAB: nfmax = round(fmax/df) + 1  (1-based, points to fmax row)
    nfmax   = int(round(fmax / df)) + 1          # last row index + 1 (exclusive slice)
    nfmax   = min(nfmax, phi.shape[0])            # clamp to available rows

    # f1 = f[1:nfmax]  (skip DC, keep up to fmax)
    # phi = phi[1:nfmax, :]
    # Then nfmax is reassigned to nfmax-1 (new length after dropping DC)
    f1  = f[1:nfmax]
    phi = phi[1:nfmax, :].copy()
    nfmax = nfmax - 1    # new nfmax = number of usable freq rows

    f2  = f1 ** 2
    nkxpad = phi.shape[1]

    print(f"{nz} depth steps")

    seismig = np.zeros((nz, nx))

    t_start = time.time()

    for iz in range(nz):
        # Nearest depth index in velocity model
        izv = int(near(zv, z[iz])[0])

        # Exploding reflector: halve velocity
        v  = 0.5 * velmod[izv, :]   # shape (ntr_vel,)
        vm = float(np.mean(v))

        # --------------------------------------------------------------
        # Loop over kx columns: focusing phase shift
        # --------------------------------------------------------------
        for jk in range(nkxpad):
            # First non-evanescent frequency for this kx
            # MATLAB: fev = kx(jk)*vm; nfev = max(round(fev/df), 1)
            fev  = float(kx[jk]) * vm           # can be negative -> nfev=0
            nfev = max(int(round(fev / df)), 0)  # 0-based index into f1

            if nfev <= nfmax - 1:
                # Focusing phase shift for propagating frequencies
                # f1_slice = f1[nfev:nfmax]  (0-based)
                f1_sl = f1[nfev:]
                f2_sl = f2[nfev:]
                psf = (
                    (2.0 * np.pi * dz * f1_sl / vm)
                    * (np.sqrt(
                        np.maximum(1.0 - vm * vm * kx2[jk] / f2_sl, 0.0)
                    ) - 1.0)
                )
                phi[nfev:, jk] *= np.exp(1j * psf)
                if nfev > 0:
                    phi[:nfev, jk] = 0.0
            else:
                # All frequencies are evanescent for this kx
                phi[:, jk] = 0.0

        # --------------------------------------------------------------
        # Interpolate v to x grid if needed
        # (MATLAB: if length(v)~=length(x) -> pwlint(xv, v, x))
        # velmod columns match x columns per the docstring requirement
        # but guard anyway using pwlint if sizes differ
        # --------------------------------------------------------------
        if len(v) != nx:
            # velmod x-coordinates assumed uniform over same span as x
            xv = np.linspace(x[0], x[-1], len(v))
            v  = pwlint(xv, v, x)

        # --------------------------------------------------------------
        # Loop over frequency rows: static phase shift + imaging
        # --------------------------------------------------------------
        for jf in range(nfmax):
            # fft of first nx columns of this freq row (kx→x)
            tmp = np.fft.fft(phi[jf, :nx])

            # Static shift using f[jf] (0-based into original f array)
            # Matches MATLAB f(jf) where jf is 1-based → f[jf-1]=f[jf-1]
            # In Python: jf=0..nfmax-1, use f[jf] (0-based original f)
            # f[0]=0 Hz (DC), so jf=0 gets no shift – matches MATLAB
            static = np.exp(1j * 2.0 * np.pi * f[jf] * dz / v)
            tmp   *= static

            # Imaging: accumulate
            seismig[iz, :] += 2.0 * np.real(tmp)

            # Store back: ifft(tmp) + zero-pad to nkxpad
            phi_row = np.fft.ifft(tmp)
            phi[jf, :] = np.concatenate(
                [phi_row, np.zeros(nkxpad - nx)]
            )

        if (iz + 1) % max(1, nz // 10) == 0 or iz == nz - 1:
            elapsed = time.time() - t_start
            print(f" finished step {iz + 1} of {nz}  ({elapsed:.1f} s)")

    zmig = z
    total = time.time() - t_start
    print(f"finished in {total:.2f} seconds")
    return seismig, zmig
