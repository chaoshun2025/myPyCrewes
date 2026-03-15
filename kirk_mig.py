"""
kirk_mig.py  –  Full-featured Kirchhoff time migration.

Ports the CREWES MATLAB function kirk_mig.m to Python/NumPy.
Helper utilities (cos_taper, sinci) are imported from their own modules;
lin_taper and near are implemented inline here.
"""

import numpy as np
import time
from cos_taper import cos_taper
from sinci import sinci


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lin_taper(start: float, end: float) -> np.ndarray:
    """
    Linear taper from 1 (at ``start``) to 0 (at ``end``).
    Returns an array of length ``round(end - start) + 1``.
    """
    n = int(round(abs(end - start))) + 1
    return np.linspace(1.0, 0.0, n)


def _near(vec: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Return indices of vec where vmin <= vec[i] <= vmax."""
    return np.where((vec >= vmin) & (vec <= vmax))[0]


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def kirk_mig(
    aryin: np.ndarray,
    aryvel,
    dt,
    dx,
    params=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full-featured Kirchhoff time migration.

    Parameters
    ----------
    aryin : np.ndarray, shape (nsamp, ntr)
        Zero-offset data matrix (one trace per column).
    aryvel : scalar, 1D array, or 2D array
        Velocity (see detailed description below).
    dt : scalar or 1D array
        Time sample interval (s) or time coordinate vector.
    dx : scalar or 1D array
        Spatial sample interval or x coordinate vector.
    params : array-like, length up to 12, optional
        Migration control parameters. Use NaN for defaults.
        [0]  aperture length (m or ft)           default: full section
        [1]  aperture taper width                default: aperture/20
        [2]  0=linear taper, 1=cosine taper      default: 1
        [3]  max dip angle (deg)                 default: 60
        [4]  angle-taper width (deg)             default: 0.15*params[3]
        [5]  0=linear, 1=cosine angle taper      default: 1
        [6]  interpolation: 1=linear,2=spline,   default: 1
             3=cubic,4=sinc
        [7]  tmin of target window (s)           default: 0
        [8]  tmax of target window (s)           default: max(t)
        [9]  xmin of target window               default: min(x)
        [10] xmax of target window               default: max(x)
        [11] boxcar anti-aliasing filter          default: 0 (off)

    Returns
    -------
    arymig : np.ndarray
        Migrated time section.
    tmig : np.ndarray
        Time coordinates of migrated data.
    xmig : np.ndarray
        X coordinates of migrated data.
    """
    aryin = np.asarray(aryin, dtype=float)
    nsamp, ntr = aryin.shape

    # ---- time axis ----
    dt_arr = np.atleast_1d(dt)
    if len(dt_arr) > 1:
        if len(dt_arr) != nsamp:
            raise ValueError("Length of t vector must equal number of rows in aryin.")
        t = dt_arr.astype(float).ravel()
        dt_val = t[1] - t[0]
    else:
        dt_val = float(dt)
        t = np.arange(nsamp) * dt_val

    # ---- x axis ----
    dx_arr = np.atleast_1d(dx)
    if len(dx_arr) > 1:
        if len(dx_arr) != ntr:
            raise ValueError("Length of x vector must equal number of columns in aryin.")
        x = dx_arr.astype(float).ravel()
        dx_val = x[1] - x[0]
    else:
        dx_val = float(dx)
        x = np.arange(ntr) * dx_val

    # ---- velocity matrix ----
    aryvel = np.asarray(aryvel, dtype=float)
    if aryvel.ndim == 0 or aryvel.size == 1:
        vel_mat = float(aryvel) * np.ones((nsamp, ntr))
    elif aryvel.ndim == 1:
        v = aryvel.ravel()
        if len(v) == nsamp:
            vel_mat = v[:, None] * np.ones((1, ntr))
        elif len(v) == ntr:
            vel_mat = np.ones((nsamp, 1)) * v[None, :]
        else:
            raise ValueError("Velocity vector has wrong size.")
    else:
        if aryvel.shape[0] != nsamp:
            raise ValueError("Velocity matrix has wrong number of rows.")
        if aryvel.shape[1] != ntr:
            raise ValueError("Velocity matrix has wrong number of columns.")
        vel_mat = aryvel

    # ---- default parameters ----
    nparams = 12
    if params is None:
        params = [np.nan] * nparams
    else:
        params = list(params)
    while len(params) < nparams:
        params.append(np.nan)
    params = [np.nan if (p is None or (isinstance(p, float) and np.isnan(p))) else p
              for p in params]

    def _p(i, default):
        v = params[i]
        return default if (v is np.nan or (isinstance(v, float) and np.isnan(float(v)))) else v

    aper      = _p(0,  abs(x[-1] - x[0]))
    width1    = _p(1,  aper / 20.0)
    itaper1   = int(_p(2, 1))
    ang_limit = _p(3, 60.0) * np.pi / 180.0
    width2    = _p(4, 0.15 * _p(3, 60.0)) * np.pi / 180.0
    angle1    = ang_limit + width2
    itaper2   = int(_p(5, 1))
    interp_type = int(_p(6, 1))
    tmig1     = _p(7,  t.min())
    tmig2     = _p(8,  t.max())
    xmig1     = _p(9,  x.min())
    xmig2     = _p(10, x.max())
    ibcfilter = int(_p(11, 0))

    # ---- aperture taper ----
    traper0 = 0.5 * aper / dx_val
    traper1 = width1 / dx_val
    traper  = int(round(traper0 + traper1))
    if itaper1 == 0:
        coef1 = _lin_taper(traper0, traper0 + traper1)
    else:
        coef1 = cos_taper(traper0, traper0 + traper1)

    # ---- one-way time axis ----
    dt1 = 0.5 * dt_val
    t1  = t / 2.0
    t2  = t1 ** 2

    # ---- maximum needed time ----
    vmin = vel_mat.min()
    tmax = np.sqrt(0.25 * tmig2 ** 2 + ((0.5 * aper + width1) / vmin) ** 2)

    # ---- pad aryin ----
    npad = int(np.ceil(tmax / dt1)) - nsamp + 5
    if npad > 0:
        aryin_pad = np.vstack([aryin, np.zeros((npad, ntr))])
        t1_pad = np.concatenate([t1, (nsamp + np.arange(npad)) * dt1])
    else:
        aryin_pad = aryin
        t1_pad = t1

    if ibcfilter:
        arycum = np.cumsum(aryin_pad, axis=0)

    # ---- target windows ----
    samptarget = _near(t, tmig1, tmig2)
    tmig       = t[samptarget]
    trtarget   = _near(x, xmig1, xmig2)
    xmig       = x[trtarget]

    arymig = np.zeros((len(samptarget), len(trtarget)))

    # ---- migration loop ----
    print(f"\n --- Total number of traces to be migrated : {len(trtarget)} ---\n")
    clock1 = time.time()
    kmig = 0

    for ktr in trtarget:
        kmig += 1

        n1 = max(0, ktr - traper)
        n2 = min(ntr - 1, ktr + traper)
        truse = np.arange(n1, n2 + 1)

        offset2 = ((truse - ktr) * dx_val) ** 2
        v2 = vel_mat[:, ktr] ** 2  # shape (nsamp,)

        if kmig % 20 == 0:
            elapsed = time.time() - clock1
            remaining = (elapsed / kmig) * (trtarget[-1] - ktr)
            print(f" Migrated trace no. {kmig} of {len(trtarget)}, "
                  f"traces in aperture: {len(truse)}")
            print(f"Estimated time remaining {remaining:.0f} seconds")

        for ki, kaper_tr in enumerate(truse):
            # Diffraction traveltime for each output sample
            t_aper = np.sqrt(
                offset2[ki] / v2[samptarget] + t2[samptarget]
            )  # shape (len(samptarget),)

            # Cosine-of-angle amplitude correction
            costheta = 0.5 * tmig / (t_aper + 100 * np.finfo(float).eps)
            if len(costheta) > 0:
                costheta[0] = 1.0

            tanalpha = np.sqrt(np.clip(1.0 - costheta ** 2, 0, None))

            # Angle taper
            ind_outer = np.where(costheta < np.cos(angle1))[0]
            if len(ind_outer) > 0:
                i1 = ind_outer[-1]
                ind_inner = np.where(costheta < np.cos(ang_limit))[0]
                if len(ind_inner) > 0:
                    i2 = ind_inner[-1]
                    if i1 < i2:
                        if itaper2 == 0:
                            coef2 = _lin_taper(i2, i1)
                        else:
                            coef2 = cos_taper(i2, i1)
                        costheta[:i1 + 1] = 0.0
                        n_taper = i2 - i1
                        costheta[i1 + 1: i2 + 1] *= coef2[n_taper - 1:: -1][:n_taper]

            # Anti-aliasing boxcar filter
            if ibcfilter:
                lt0 = np.round(
                    dx_val * tanalpha / vel_mat[samptarget, ktr] / dt1
                ).astype(int)
                indt = np.round(t_aper / dt1).astype(int)
                lentr = len(t1_pad)
                lt = np.ones(lentr, dtype=int) * lt0.max()
                lt[indt] = lt0
                lt[indt.max() + 1:] = lt0.min()
                it = np.arange(lentr)
                l1 = np.clip(it - lt - 1, 0, lentr - 1)
                l2 = np.clip(it + lt, 0, lentr - 1)
                tmp0 = np.empty(lentr)
                tmp0[0] = arycum[0, kaper_tr]
                dlen = l2[1:] - l1[1:]
                tmp0[1:] = np.where(
                    dlen > 0,
                    (arycum[l2[1:], kaper_tr] - arycum[l1[1:], kaper_tr]) / dlen,
                    0.0,
                )
            else:
                tmp0 = aryin_pad[:, kaper_tr]

            # Interpolation
            if interp_type == 1:
                tnumber = t_aper / dt1
                it0 = np.floor(tnumber).astype(int)
                it1 = it0 + 1
                xt0 = tnumber - it0
                xt1 = 1.0 - xt0
                it0 = np.clip(it0, 0, len(tmp0) - 1)
                it1 = np.clip(it1, 0, len(tmp0) - 1)
                tmp = xt1 * tmp0[it0] + xt0 * tmp0[it1]
            elif interp_type == 2:
                tmp = np.interp(t_aper, t1_pad, tmp0, left=tmp0[0], right=tmp0[-1])
            elif interp_type == 3:
                from scipy.interpolate import interp1d
                f = interp1d(t1_pad, tmp0, kind='cubic', bounds_error=False,
                             fill_value=(tmp0[0], tmp0[-1]))
                tmp = f(t_aper)
            elif interp_type == 4:
                tmp = sinci(tmp0, t1_pad, t_aper)
            else:
                raise ValueError(f"Unknown interpolation type: {interp_type}")

            # Aperture taper
            offset_tr = abs(kaper_tr - ktr) * dx_val
            if offset_tr > 0.5 * aper:
                idx = int(round(abs(kaper_tr - ktr) - traper0))
                idx = np.clip(idx, 0, len(coef1) - 1)
                ccoef = coef1[idx]
                if abs(1.0 - ccoef) > 0.05:
                    tmp = tmp * ccoef

            # Amplitude weighting by cos^1.5(theta)
            ind_apply = np.where(costheta < 0.999)[0]
            costheta_mod = costheta.copy()
            costheta_mod[ind_apply] = np.sqrt(costheta[ind_apply] ** 3)
            tmp[ind_apply] *= costheta_mod[ind_apply]

            arymig[:, kmig - 1] += tmp

        # Scaling and 45-degree phase shift (scaling only — phase shift
        # should be applied separately via conv45 if desired)
        scalemig = vel_mat[samptarget, kmig - 1] * np.sqrt(
            np.pi * (tmig + 1e-4)
        )
        arymig[:, kmig - 1] /= scalemig

    elapsed = time.time() - clock1
    print(f"Migration completed in {elapsed:.0f} seconds")

    return arymig, tmig, xmig
