"""
afd_explode.py — Acoustic finite-difference exploding-reflector modelling.

Translated from MATLAB afd_explode.m (CREWES / Margrave).

This module ties together all helper functions to simulate seismic data
via the exploding-reflector concept: the velocity model is halved (one-way
travel time), the reflectivity is initialised as the initial wavefield, and
the wavefield is propagated forward in time using an explicit acoustic
finite-difference scheme.

Usage
-----
>>> from afd_explode import afd_explode
>>> seisw, seis, t = afd_explode(dx, dtstep, dt, tmax,
...                               velocity, xrec, zrec,
...                               wlet, tw,
...                               laplacian=1, boundary=2)
"""

import time
import numpy as np
from afd_reflect import afd_reflect
from afd_snap    import afd_snap
from resamp      import resamp
from convz       import convz
from filtf       import filtf


# ---------------------------------------------------------------------------
# Utility: find nearest sample index (MATLAB's near())
# ---------------------------------------------------------------------------
def _near(t: np.ndarray, *args) -> np.ndarray:
    """
    Return indices of samples closest to one or two target values.

    _near(t, t0)          → index of sample nearest t0
    _near(t, t0, t1)      → indices of all samples in [t0, t1] (inclusive)
    """
    t = np.asarray(t)
    if len(args) == 1:
        return np.array([np.argmin(np.abs(t - args[0]))])
    else:
        t0, t1 = args
        return np.where((t >= t0) & (t <= t1))[0]


# ---------------------------------------------------------------------------
# Margrave half-Hanning taper (mwhalf)
# ---------------------------------------------------------------------------
def _mwhalf(n: int, pct: float) -> np.ndarray:
    """
    One-sided cosine taper: rises from 0 to 1 over the first *pct* percent
    of the window, then stays at 1.  Used to suppress shallow reflectivity.
    """
    m = max(1, int(round(pct * n / 100.0)))
    taper = 0.5 * (1.0 - np.cos(np.pi * np.arange(m) / (m - 1)))
    w = np.ones(n)
    w[:m] = taper
    return w


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def afd_explode(dx:        float,
                dtstep:    float,
                dt:        float,
                tmax:      float,
                velocity:  np.ndarray,
                xrec:      np.ndarray,
                zrec:      np.ndarray,
                wlet,               # Ormsby spec [f1,f2,f3,f4] or wavelet array
                tw,                 # phase flag (0/1) or time vector for wavelet
                laplacian: int = 1,
                boundary:  int = 2,
                zmin:      float = 0.0
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a seismogram using the exploding-reflector method.

    Parameters
    ----------
    dx        : float
        Spatial bin spacing (m) — used for both x and z.
    dtstep    : float
        Finite-difference time step (s).  Must satisfy the stability
        condition for the chosen Laplacian.
    dt        : float
        Output sample interval (s).  ``abs(dt) >= dtstep``.
        Sign controls anti-alias filter phase:
        positive → minimum phase, negative → zero phase.
    tmax      : float
        Maximum record length (s).
    velocity  : ndarray, shape (nz, nx)
        P-wave velocity model (m/s).  **Do not halve** — halving is done
        internally to convert to one-way travel time.
    xrec      : array-like
        x-positions of receivers (m).
    zrec      : array-like
        z-positions of receivers (m).  Use 0 for surface receivers.
    wlet      : array-like
        Either a 4-element Ormsby specification ``[f1, f2, f3, f4]`` (Hz) or
        a wavelet time series.
    tw        : scalar or array-like
        If *wlet* is Ormsby: phase flag (``0`` = zero-phase, ``1`` = min-phase).
        If *wlet* is a wavelet: time coordinate vector for that wavelet.
        **The wavelet sample rate must equal ``abs(dt)``.**
    laplacian : int, optional
        ``1`` → 5-point Laplacian (default, faster).
        ``2`` → 9-point Laplacian (broader bandwidth).
    boundary  : int, optional
        ``0`` no absorbing BCs, ``1`` all four sides, ``2`` three sides
        (top free).  Default 2.
    zmin      : float, optional
        Reflectivity shallower than this depth is suppressed.  Default 0.

    Returns
    -------
    seisw : ndarray, shape (nt, nrec)
        Seismogram convolved / filtered with the wavelet.
    seis  : ndarray, shape (nt, nrec)
        Raw (unfiltered) seismogram.
    t     : ndarray, shape (nt,)
        Time vector (s).

    Stability conditions
    --------------------
    * 5-point:  ``max(velocity) * dtstep / dx  <  1/sqrt(2) ≈ 0.707``
    * 9-point:  ``max(velocity) * dtstep / dx  <  sqrt(3/8) ≈ 0.612``
    """
    t_start = time.time()

    # ------------------------------------------------------------------ setup
    velocity = np.asarray(velocity, dtype=float)
    xrec     = np.asarray(xrec,     dtype=float).ravel()
    zrec     = np.asarray(zrec,     dtype=float).ravel()
    wlet     = np.asarray(wlet,     dtype=float).ravel()

    nz, nx = velocity.shape
    x = np.arange(nx) * dx
    z = np.arange(nz) * dx
    xmax = x[-1];  zmax = z[-1]

    # Parse wavelet / filter specification
    ormsby_mode = False
    if len(wlet) == 4:
        # Ormsby spec: convert to filtf-style [f2, f2-f1, f3, f4-f3]
        f1, f2, f3, f4 = wlet
        filt_spec = [f2, f2 - f1, f3, f4 - f3]
        if tw not in (0, 1):
            raise ValueError("For Ormsby spec, tw must be 0 (zero-phase) "
                             "or 1 (minimum-phase).")
        phase = int(tw)
        ormsby_mode = True
    else:
        print('len(tw) is ', len(tw))
        print('len(wlet)is ', len(wlet))
        tw = np.asarray(tw, dtype=float).ravel()
        if len(wlet) != len(tw):
            raise ValueError("wlet and tw must have the same length.")
        dt_wav = abs(tw[1] - tw[0])
        if abs(dt_wav - abs(dt)) > 1e-9:
            raise ValueError("The wavelet sample rate must equal abs(dt).")
        w = wlet.copy()

    if abs(dt) < dtstep:
        raise ValueError("abs(dt) cannot be less than dtstep.")

    # ---------------------------------------------------------------- stability
    vel_half = velocity / 2.0          # exploding-reflector: halve velocity
    vmax     = np.max(vel_half)
    courant  = vmax * dtstep / dx
    if laplacian == 1 and courant > 1.0 / np.sqrt(2):
        raise RuntimeError(
            f"Simulation unstable: max(v/2)*dtstep/dx = {courant:.4f} "
            f"must be < {1/np.sqrt(2):.4f}")
    if laplacian == 2 and courant > np.sqrt(3.0 / 8.0):
        raise RuntimeError(
            f"Simulation unstable: max(v/2)*dtstep/dx = {courant:.4f} "
            f"must be < {np.sqrt(3/8):.4f}")

    # -------------------------------------------------- initial reflectivity
    clipn = 0
    snap2 = afd_reflect(velocity, clipn)

    if zmin > 0 and zmax > 0:
        pct  = 100.0 * zmin / zmax
        mw   = np.flipud(_mwhalf(len(z), pct))
        snap2 = snap2 * mw[:, np.newaxis]

    snap1 = np.zeros_like(snap2)

    # ----------------------------------------------- receiver index mapping
    nrec = len(xrec)
    xrec_idx = (np.floor((xrec - x[0]) / dx)).astype(int)
    zrec_idx = (np.floor((zrec - z[0]) / dx)).astype(int)
    # Clamp to valid range
    xrec_idx = np.clip(xrec_idx, 0, nx - 1)
    zrec_idx = np.clip(zrec_idx, 0, nz - 1)

    # ------------------------------------------------------- time-step loop
    maxstep  = int(round(tmax / dtstep))
    seis     = np.zeros((maxstep + 1, nrec))
    seis[0]  = snap2[zrec_idx, xrec_idx]

    print(f"There are {maxstep} steps to complete.")
    t0       = time.time()
    nwrite   = max(1, 2 * round(maxstep / 50) + 1)

    for k in range(1, maxstep + 1):
        snapshot, _, _ = afd_snap(dx, dtstep, vel_half,
                                   snap1, snap2, laplacian, boundary)
        seis[k] = snapshot[zrec_idx, xrec_idx]
        snap1   = snap2
        snap2   = snapshot

        if k % nwrite == 0:
            elapsed  = time.time() - t0
            remaining = elapsed * maxstep / k - elapsed
            print(f"  wavefield propagated to {k*dtstep:.4f} s; "
                  f"time left ≈ {remaining:.1f} s")

    print("Modelling completed.")

    # --------------------------------------------------------- time vector
    t = np.arange(seis.shape[0]) * dtstep

    # --------------------------------------------------------- resampling
    if abs(dt) != dtstep:
        print("Resampling...")
        phs  = 1 if dt > 0 else 0
        dt_  = abs(dt)
        seis_new = None
        t2        = None
        for k in range(nrec):
            cs = np.polyfit(t, seis[:, k], 4)
            tmp_detrend = seis[:, k] - np.polyval(cs, t)
            tmp_rs, t2  = resamp(tmp_detrend, t, dt_,
                                 [t[0], t[-1]], phs)
            tmp_rs += np.polyval(cs, t2)
            if seis_new is None:
                seis_new = np.zeros((len(tmp_rs), nrec))
            seis_new[:, k] = tmp_rs
        seis = seis_new
        t    = t2

    # -------------------------------------------------- wavelet convolution
    seisw = np.zeros_like(seis)
    nt    = len(t)
    dt_   = abs(dt) if abs(dt) != dtstep else dtstep

    if not ormsby_mode:
        # wavelet convolution via convz
        nzero = _near(tw, 0)[0] + 1    # 1-based index of zero-time sample
        print("Applying wavelet...")
        ifit = _near(t, 0.9 * t[-1], t[-1])
        tpad = np.arange(t[-1], 1.5 * t[-1] + dt_, dt_)[1:]

        for k in range(nrec):
            tmp = seis[:, k]
            cs  = np.polyfit(t[ifit], tmp[ifit], 1)
            tmp = np.concatenate([tmp, np.polyval(cs, tpad)])
            tmp2 = convz(tmp, w, nzero)
            seisw[:, k] = tmp2[:nt]
    else:
        # Ormsby / filtf path
        print("Filtering...")
        ifit = _near(t, 0.9 * t[-1], t[-1])
        tpad = np.arange(t[-1] + dt_, 1.1 * t[-1] + dt_, dt_)

        for k in range(nrec):
            tmp = seis[:, k]
            cs  = np.polyfit(t[ifit], tmp[ifit], 1)
            tmp_ext = np.concatenate([tmp, np.polyval(cs, tpad)])
            t_ext   = np.concatenate([t, tpad])
            tmp2    = filtf(tmp_ext, t_ext,
                            [filt_spec[0], filt_spec[1]],
                            [filt_spec[2], filt_spec[3]],
                            phase)
            seisw[:, k] = tmp2[:nt]

    if np.iscomplexobj(seisw):
        seisw = np.real(seisw)

    print(f"Total elapsed: {time.time() - t_start:.1f} s")
    return seisw, seis, t
