"""pspi_stack.py – zero-offset section depth migration via the PSPI algorithm.

Mirrors the MATLAB ``pspi_stack`` from the Ferguson / Margrave CREWES toolbox.

Reference
---------
Ferguson & Margrave (2005), "Planned seismic imaging using explicit one-way
operators", *Geophysics*, 70(5), S101–S109.
"""

import time
import numpy as np

from near import near
from fktran import fktran
from ifktran import ifktran
from pspi_ips import pspi_ips
from ps_rezero import ps_rezero
from Bagaini import Bagaini


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _zos2t(zos_fk, f, dx):
    """
    Convert a one-sided f-k spectrum back to the t-x domain.

    Parameters
    ----------
    zos_fk : np.ndarray (complex), shape (nf, nx)
        f-k spectrum with k axis wrapped and positive f's only.
    f : array_like
        Frequency vector (Hz).
    dx : float
        Spatial sample interval (m).

    Returns
    -------
    zos_t : np.ndarray (float), shape (nt, nx)
        Time-domain section.
    """
    f    = np.asarray(f, dtype=float).ravel()
    nc   = zos_fk.shape[1]
    df   = f[1] - f[0]
    fmax = f[-1]

    # Find dt such that fnyq >= fmax
    dt   = 0.008
    fnyq = 0.5 / dt
    while fnyq < fmax:
        dt   /= 2.0
        fnyq  = 0.5 / dt

    fnew   = np.arange(0.0, fnyq + df * 0.5, df)
    nf_new = len(fnew)

    kx = np.fft.fftshift(
        np.arange(-nc, nc, 2, dtype=float) / (2.0 * nc * dx)
    )

    indf = near(fnew, f[0], f[-1])
    data = np.zeros((nf_new, nc), dtype=complex)
    data[indf, :] = zos_fk
    data[-1, :]   = 0.0      # real Nyquist

    zos_t, _, _ = ifktran(data, fnew, kx, 0, 0)
    return zos_t


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def pspi_stack(zos, t, x, vel, xv, zv,
               frange=None, zcheck=None, irezero=5):
    """
    2-D depth migration of a zero-offset section (ZOS) via the PSPI algorithm.

    Parameters
    ----------
    zos : np.ndarray, shape (nt, nx)
        Zero-offset section; one trace per column.
    t : array_like, length nt
        Time coordinate vector (s) for *zos*.
    x : array_like, length nx
        Lateral coordinate vector (m) for *zos* (regularly sampled).
    vel : np.ndarray, shape (nz, nx_vel)
        P-wave interval velocity model (m/s).  Each row is a constant depth.
    xv : array_like, length nx_vel
        Lateral coordinate vector (m) for *vel*.
    zv : array_like, length nz
        Depth coordinate vector (m) for *vel* (regularly sampled).
    frange : array_like of length 2, optional
        ``[fmin, fmax]`` frequency range (Hz) to migrate.
        Default: all frequencies ``[0, inf]``.
    zcheck : array_like, optional
        Depths (m) at which to save the extrapolated ZOS.
        Default: ``[]`` (no snapshots saved).
    irezero : int, optional
        Re-zero the temporal pad every this many depth steps.
        Default: ``5``.

    Returns
    -------
    zosmig : np.ndarray, shape (nz, nx_vel)
        Depth-migrated image.  Same size as the velocity model.
    exzos : list of np.ndarray
        Time-domain extrapolated sections at each depth in *zcheck*.
        Empty list if *zcheck* is empty.

    Notes
    -----
    * The ZOS is automatically zero-padded in x to match the velocity model
      extent, and in t to a power-of-2 length sufficient to hold at least
      10 depth steps of propagation before wrap-around.
    * The velocity model is halved (exploding-reflector convention) before
      migration.
    * Progress messages are printed every 30 depth steps and whenever the
      temporal pad is re-zeroed.
    """
    # ------------------------------------------------------------------ #
    # Setup and validation
    # ------------------------------------------------------------------ #
    zos = np.asarray(zos, dtype=float)
    t   = np.asarray(t,   dtype=float).ravel()
    x   = np.asarray(x,   dtype=float).ravel()
    vel = np.asarray(vel, dtype=float)
    xv  = np.asarray(xv,  dtype=float).ravel()
    zv  = np.asarray(zv,  dtype=float).ravel()

    if frange is None:
        frange = [0.0, np.inf]
    if zcheck is None:
        zcheck = np.array([], dtype=float)
    zcheck = np.asarray(zcheck, dtype=float).ravel()

    Nz, Nx = vel.shape
    dz     = zv[1] - zv[0]
    nt, nx = zos.shape
    dt     = t[1] - t[0]
    small  = 1e-4
    dx     = xv[1] - xv[0]
    dxs    = x[1]  - x[0]

    # Dimension checks
    if len(x) != nx:
        raise ValueError("zos x axis length does not match zos columns")
    if len(t) != nt:
        raise ValueError("zos t axis length does not match zos rows")
    if len(zv) != Nz:
        raise ValueError("velocity z axis incorrect")
    if len(xv) != Nx:
        raise ValueError("velocity x axis incorrect")
    if abs(dxs - dx) > small:
        raise ValueError("velocity model and zos must have same x sample size")

    # ZOS x-coordinates must lie within velocity model extent
    xmins, xmaxs = x.min(), x.max()
    xminv, xmaxv = xv.min(), xv.max()
    if xmins < xminv or xmaxs > xmaxv:
        raise ValueError(
            "zos x coordinates fall outside the span of the velocity model. "
            "Velocity model must be extended."
        )
    if abs(dx * np.floor(xmaxs / dx) - xmaxs) > small:
        raise ValueError(
            "Velocity model and zos record are not on the same x grid"
        )
    if np.sum(np.abs(np.diff(np.diff(x)))) > small:
        raise ValueError("zos record must be regularly sampled in x")

    # ------------------------------------------------------------------ #
    # Exploding-reflector convention: halve the velocity
    # ------------------------------------------------------------------ #
    vel = vel / 2.0

    # ------------------------------------------------------------------ #
    # Pad zos laterally to match velocity model extent
    # ------------------------------------------------------------------ #
    npadmin = 0
    npadmax = 0
    if xmins > xminv:
        npadmin = int(round((xmins - xminv) / dx))
    if xmaxs < xmaxv:
        npadmax = int(round((xmaxv - xmaxs) / dx))
    if npadmin + npadmax > 0:
        zos = np.hstack([
            np.zeros((nt, npadmin)),
            zos,
            np.zeros((nt, npadmax))
        ])
    x  = xv.copy()
    nx = len(x)

    # Pad laterally to next power of 2
    nx2 = int(2 ** np.ceil(np.log2(nx)))
    if nx < nx2:
        zos = np.hstack([zos, np.zeros((nt, nx2 - nx))])
        vel = np.hstack([vel, vel[:, -1:] * np.ones((Nz, nx2 - nx))])
    x2 = np.arange(nx2) * dx

    # ------------------------------------------------------------------ #
    # Pad zos temporally to next power of 2
    # ------------------------------------------------------------------ #
    tmax  = t.max()
    vmin  = vel.min()
    tpad  = 2.0 * 10 * dz / vmin         # two-way time for 10 steps at vmin
    npad  = int(round(tpad / dt))
    npow  = int(np.ceil(np.log2(nt + npad)))
    ntnew = int(2 ** npow)
    npad  = ntnew - nt
    zos   = np.vstack([zos, np.zeros((npad, nx2))])
    t_pad = dt * np.arange(ntnew)

    # ------------------------------------------------------------------ #
    # Forward f-k transform
    # ------------------------------------------------------------------ #
    zosfk, f, kx = fktran(zos, t_pad, x2)
    # pspi_ips expects a wrapped kx spectrum – undo the fftshift
    zosfk = np.fft.ifftshift(zosfk, axes=1)

    # Frequency range
    if frange[0] == 0.0:
        frange[0] = f[1]          # skip DC
    if frange[1] > f[-1]:
        frange[1] = f[-1]

    indf = near(f, frange[0], frange[1])
    nf2  = len(indf)

    # ------------------------------------------------------------------ #
    # Build blocked velocity model (Bagaini method)
    # ------------------------------------------------------------------ #
    vel_blocked = Bagaini(len(x) - 1, 10, vel)

    # ------------------------------------------------------------------ #
    # Migration loop
    # ------------------------------------------------------------------ #
    zosmig = np.zeros((Nz, nx2))

    # Remove zcheck depths beyond the model
    zcheck = zcheck[zcheck < zv.max()]

    exzos  = []
    jcheck = -1
    ncheck = 0
    if len(zcheck) > 0:
        jcheck = int(round(zcheck[0] / dz))   # 0-based depth index

    time1    = time.time()
    timeused = 0.0

    for j in range(Nz - 1):
        # ---- progress and periodic re-zero ----
        if (j + 1) % 30 == 0:
            timeremaining = (Nz - 1) * timeused / (j + 1) - timeused
            print(
                f" pspi_stack: depth {j + 1} of {Nz}, "
                f"time left ~ {int(timeremaining)} s"
            )
        elif (j + 1) % irezero == 0:
            print(f" pspi_stack: depth {j + 1} of {Nz} – re-zeroing pad")
            zosfk = ps_rezero(zosfk, f, dx, tmax)

        # ---- one depth step ----
        ftemp = pspi_ips(
            zosfk[indf, :],
            f[indf],
            dx,
            vel[j, :],
            vel_blocked[j, :],
            dz
        )
        # ftemp is in the (f, x) domain

        # ---- imaging condition ----
        # MATLAB: real(sum(ftemp) + sum(ftemp(1:nf2-1,:))) / (2*nf2 - 1)
        # sum over frequency axis (axis=0)
        zosmig[j + 1, :] = np.real(
            ftemp.sum(axis=0) + ftemp[:nf2 - 1, :].sum(axis=0)
        ) / (2 * nf2 - 1)

        # ---- transform back to (f, kx) ----
        zosfk[indf, :] = np.fft.fft(ftemp, axis=1)

        # ---- optional extrapolated-section snapshot ----
        if j == jcheck:
            exzos.append(_zos2t(zosfk, f, dx))
            ncheck += 1
            if ncheck < len(zcheck):
                jcheck = int(round(zcheck[ncheck] / dz))

        timenow   = time.time()
        timeused += timenow - time1
        time1     = timenow

    # Remove lateral zero-pad from migrated image
    zosmig = zosmig[:, :nx]
    print(f"zos migrated in {int(timeused)} s")

    return zosmig, exzos
