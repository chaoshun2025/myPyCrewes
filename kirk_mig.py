"""
kirk_mig.py  –  Full-featured Kirchhoff time migration.

Ports the CREWES MATLAB function kirk_mig.m (Xinxiang Li / G.F. Margrave,
University of Calgary, 1996) to Python/NumPy.

Depends on the companion modules:
    cos_taper.py   – cosine taper coefficients
    near.py        – nearest-sample index search
    sinci.py       – 8-point sinc interpolation
"""

import time
import numpy as np
from scipy.interpolate import interp1d

from cos_taper import cos_taper
from near import near
from sinci import sinci


# ---------------------------------------------------------------------------
# tiny helper: linear taper (mirrors MATLAB lin_taper used in original code)
# ---------------------------------------------------------------------------
def _lin_taper(sp: float, ep: float, samp: float = 1.0) -> np.ndarray:
    """Linear taper from 1.0 at *sp* to 0.0 at *ep*."""
    if samp < 0:
        samp = -samp
    length = round(abs(ep - sp) / samp) + 1
    if length <= 1:
        return np.array([1.0])
    return np.linspace(1.0, 0.0, length)


# ---------------------------------------------------------------------------
# main function
# ---------------------------------------------------------------------------
def kirk_mig(
    aryin: np.ndarray,
    aryvel,
    dt,
    dx,
    params=None,
):
    """
    Full-featured Kirchhoff time migration.

    Parameters
    ----------
    aryin : ndarray, shape (nsamp, ntr)
        Zero-offset data matrix. One trace per column.
    aryvel : scalar | 1-D array | 2-D array
        RMS velocity information.
        - scalar  → constant velocity for every sample/trace.
        - 1-D (nsamp,) → velocity function of time, applied uniformly.
        - 2-D (nsamp, ntr) → velocity at every sample location.
    dt : scalar or 1-D array
        If scalar: time sample rate in seconds.
        If array of length nsamp: the time coordinate vector.
    dx : scalar or 1-D array
        If scalar: spatial sample interval (same units as velocity).
        If array of length ntr: x-coordinate vector for each trace.
    params : array_like of length ≤ 12, optional
        Migration parameters (NaN → use default).
        [0]  aperture length              (default: section length)
        [1]  aperture taper width         (default: aperture/20)
        [2]  aperture taper type 0=lin, 1=cos  (default: 1)
        [3]  max dip angle (degrees)      (default: 60)
        [4]  angle-limit taper width (deg)(default: 0.15*params[3])
        [5]  angle taper type 0=lin, 1=cos (default: 1)
        [6]  interpolation 1=linear, 2=cubic, 3=spline, 4=sinc (default: 1)
        [7]  tmin of output window        (default: min(t))
        [8]  tmax of output window        (default: max(t))
        [9]  xmin of output window        (default: min(x))
        [10] xmax of output window        (default: max(x))
        [11] boxcar anti-alias filter 0=off, 1=on  (default: 0)

    Returns
    -------
    arymig : ndarray, shape (ntmig, nxmig)
        Migrated time section.
    tmig : ndarray, shape (ntmig,)
        Time coordinates of migrated data.
    xmig : ndarray, shape (nxmig,)
        x-coordinates of migrated data.
    """

    aryin = np.asarray(aryin, dtype=float)
    nsamp, ntr = aryin.shape

    aryvel = np.asarray(aryvel, dtype=float)

    # ------------------------------------------------------------------
    # parse dt / t
    # ------------------------------------------------------------------
    dt_arr = np.asarray(dt, dtype=float).ravel()
    if dt_arr.size > 1:
        if dt_arr.size != nsamp:
            raise ValueError("Incorrect time specification")
        t = dt_arr.copy()
        dt_val = t[1] - t[0]
    else:
        dt_val = float(dt_arr[0])
        t = np.arange(nsamp) * dt_val

    # ------------------------------------------------------------------
    # parse dx / x
    # ------------------------------------------------------------------
    dx_arr = np.asarray(dx, dtype=float).ravel()
    if dx_arr.size > 1:
        if dx_arr.size != ntr:
            raise ValueError("Incorrect x specification")
        x = dx_arr.copy()
        dx_val = x[1] - x[0]
    else:
        dx_val = float(dx_arr[0])
        x = np.arange(ntr) * dx_val

    # ------------------------------------------------------------------
    # expand velocity to (nsamp, ntr)
    # ------------------------------------------------------------------
    if aryvel.ndim == 0 or aryvel.size == 1:
        aryvel = float(aryvel) * np.ones((nsamp, ntr))
    elif aryvel.ndim == 1 or min(aryvel.shape) == 1:
        v1d = aryvel.ravel()
        if v1d.size == nsamp:
            aryvel = np.outer(v1d, np.ones(ntr))
        elif v1d.size == ntr:
            # could be a spatial vector – treat as time vector (transposed in MATLAB)
            # MATLAB checks nvtr==nsamp first, so mirror that:
            raise ValueError("Velocity vector is wrong size (expected nsamp length)")
        else:
            raise ValueError("Velocity vector is wrong size")
    else:
        if aryvel.shape[0] != nsamp:
            raise ValueError("Velocity matrix has wrong number of rows")
        if aryvel.shape[1] != ntr:
            raise ValueError("Velocity matrix has wrong number of columns")

    # ------------------------------------------------------------------
    # parse params
    # ------------------------------------------------------------------
    nparams = 12
    if params is None:
        p = np.full(nparams, np.nan)
    else:
        p = np.asarray(params, dtype=float).ravel()
        if p.size < nparams:
            p = np.concatenate([p, np.full(nparams - p.size, np.nan)])

    def _p(i, default):
        return default if np.isnan(p[i]) else p[i]

    aper     = _p(0, abs(x.max() - x.min()))
    width1   = _p(1, aper / 20.0)
    itaper1  = int(_p(2, 1))
    ang_limit_deg = _p(3, 60.0)
    ang_limit = ang_limit_deg * np.pi / 180.0
    width2_deg = _p(4, 0.15 * ang_limit_deg)
    width2   = width2_deg * np.pi / 180.0
    angle1   = ang_limit + width2                # outer edge of angle taper
    itaper2  = int(_p(5, 1))
    if itaper2 not in (0, 1):
        raise ValueError("params[5] (angle taper type) must be 0 or 1")
    interp_type = int(_p(6, 1))
    if interp_type < 1 or interp_type > 4:
        raise ValueError("params[6] (interpolation type) must be 1–4")
    tmig1 = _p(7, t.min())
    tmig2 = _p(8, t.max())
    if tmig2 < tmig1:
        raise ValueError("params[7] (tmin) must be <= params[8] (tmax)")
    xmig1 = _p(9,  x.min())
    xmig2 = _p(10, x.max())
    if xmig2 < xmig1:
        raise ValueError("params[9] (xmin) must be <= params[10] (xmax)")
    ibcfilter = int(_p(11, 0))

    # ------------------------------------------------------------------
    # optional boxcar cumulative sum
    # ------------------------------------------------------------------
    if ibcfilter:
        arycum = np.cumsum(aryin, axis=0)

    # ------------------------------------------------------------------
    # aperture in trace units and taper coefficients
    # ------------------------------------------------------------------
    traper0 = 0.5 * aper / dx_val          # half-aperture in trace indices
    traper1 = width1 / dx_val              # taper width in trace indices
    traper  = int(round(traper0 + traper1))

    if itaper1 == 0:
        coef1 = _lin_taper(traper0, traper0 + traper1)
    else:
        coef1 = cos_taper(traper0, traper0 + traper1)

    # ------------------------------------------------------------------
    # one-way time axes
    # ------------------------------------------------------------------
    dt1 = 0.5 * dt_val          # one-way sample interval
    t1  = t / 2.0               # one-way time vector  (length nsamp)
    t2  = t1 ** 2

    # ------------------------------------------------------------------
    # compute maximum one-way time needed and pad aryin
    # ------------------------------------------------------------------
    vmin = aryvel.min()
    tmax_needed = np.sqrt(0.25 * tmig2 ** 2 + ((0.5 * aper + width1) / vmin) ** 2)

    npad = int(np.ceil(tmax_needed / dt1)) - nsamp + 5
    if npad > 0:
        aryin = np.vstack([aryin, np.zeros((npad, ntr))])
        extra_t1 = (np.arange(1, npad + 1) + nsamp) * dt1
        t1 = np.concatenate([t1, extra_t1])
        if ibcfilter:
            last_row = arycum[-1:, :]
            arycum = np.vstack([arycum, np.tile(last_row, (npad, 1))])

    lentr_total = nsamp + max(npad, 0)   # total rows in (possibly padded) aryin

    # ------------------------------------------------------------------
    # output index sets
    # ------------------------------------------------------------------
    samptarget = near(t, tmig1, tmig2)          # 0-based indices into t
    tmig = t[samptarget]

    trtarget = near(x, xmig1, xmig2)            # 0-based indices into x
    xmig = x[trtarget]

    nsamp_out = len(samptarget)
    ntr_out   = len(trtarget)

    arymig = np.zeros((nsamp_out, ntr_out))

    # ------------------------------------------------------------------
    # precompute squared one-way times at output samples
    # ------------------------------------------------------------------
    t2_out = t2[samptarget]         # (nsamp_out,)
    tmig_col = tmig                 # (nsamp_out,)   two-way migrated time

    print()
    print(f" --- Total number of traces to be migrated : {ntr_out} ---")
    print()

    clock_start = time.time()

    # ------------------------------------------------------------------
    # main loop over output traces
    # ------------------------------------------------------------------
    for kmig, ktr in enumerate(trtarget):
        # aperture window (0-based)
        n1 = max(0,       ktr - traper)
        n2 = min(ntr - 1, ktr + traper)
        truse = np.arange(n1, n2 + 1)          # 0-based trace indices

        # squared offsets and velocity at output trace location
        offset2 = ((truse - ktr) * dx_val) ** 2            # (len_truse,)
        v2      = aryvel[:, ktr] ** 2                        # (nsamp,) full; sliced below

        v2_out = v2[samptarget]                              # (nsamp_out,)

        if (kmig + 1) % 20 == 0:
            elapsed = time.time() - clock_start
            remaining = (elapsed / (kmig + 1)) * (ntr_out - kmig - 1)
            print(f" Migrated trace {kmig+1} of {ntr_out},"
                  f" aperture traces: {len(truse)}")
            print(f" Estimated time remaining: {int(remaining)} seconds")

        # --------------------------------------------------------------
        # loop over input traces in aperture
        # --------------------------------------------------------------
        for ka, kaper_tr in enumerate(truse):

            # hyperbolic travel-time at each output sample
            # t_aper is one-way time (two-way / 2) to the diffractor
            t_aper = np.sqrt(offset2[ka] / v2_out + t2_out)   # (nsamp_out,)

            # ---- cosine-theta amplitude correction ----
            # costheta = (two-way mig time / 2) / one-way diffraction time
            #          = tmig / (2 * t_aper)
            costheta = 0.5 * tmig_col / (t_aper + 100.0 * np.finfo(float).eps)
            costheta[0] = 1.0   # avoid NaN at t=0

            # ---- angle-limit taper ----
            # ind1: samples beyond the outer taper edge (zero zone)
            ind1 = np.where(costheta < np.cos(angle1))[0]
            if len(ind1) > 0:
                i1 = ind1[-1]   # last (deepest) sample in zero zone
                ind2 = np.where(costheta < np.cos(ang_limit))[0]
                if len(ind2) > 0:
                    i2 = ind2[-1]   # last sample in taper zone
                    if i1 < i2:
                        # build taper of length (i2 - i1): goes from 0→1
                        # MATLAB: coef2(i2-i1:-1:1) → reversed → ramp 0..1
                        taper_len = i2 - i1   # number of taper samples
                        if itaper2 == 0:
                            coef2 = _lin_taper(i2, i1)        # length taper_len+1
                        else:
                            coef2 = cos_taper(i2, i1)         # length taper_len+1
                        # coef2 length is taper_len+1 (from i2 down to i1)
                        # MATLAB: coef2(i2-i1:-1:1) picks indices [taper_len-1, ..., 0]
                        # i.e. the reversed interior (excluding the endpoint at index 0)
                        coef2_rev = coef2[taper_len - 1:: -1]  # length taper_len
                        costheta[:i1 + 1] = 0.0
                        costheta[i1 + 1: i2 + 1] *= coef2_rev[:i2 - i1]

            # ---- select input trace (possibly boxcar-filtered) ----
            if ibcfilter:
                tanalpha = np.sqrt(np.clip(1.0 - costheta ** 2, 0.0, None))
                lt0 = np.round(
                    dx_val * tanalpha / aryvel[samptarget, ktr] / dt1
                ).astype(int)
                indt = np.round(t_aper / dt1).astype(int)   # one-way sample index

                lt = np.full(lentr_total, lt0.max(), dtype=int)
                # fill at the output-sample positions
                indt_clamped = np.clip(indt, 0, lentr_total - 1)
                lt[indt_clamped] = lt0
                # beyond the deepest output sample use lt0.min()
                max_indt = indt_clamped.max()
                if max_indt + 1 < lentr_total:
                    lt[max_indt + 1:] = lt0.min()

                it = np.arange(lentr_total)
                l1 = np.clip(it - lt - 1, 0, lentr_total - 1)
                l2 = np.clip(it + lt,     0, lentr_total - 1)

                tmp0 = np.empty(lentr_total)
                tmp0[0] = arycum[0, kaper_tr]
                idx = np.arange(1, lentr_total)
                denom = (l2[idx] - l1[idx]).astype(float)
                denom[denom == 0] = 1.0
                tmp0[idx] = (arycum[l2[idx], kaper_tr]
                             - arycum[l1[idx], kaper_tr]) / denom
            else:
                tmp0 = aryin[:, kaper_tr]   # full padded column

            # ---- interpolation ----
            if interp_type == 1:
                # manual linear interpolation (matches MATLAB exactly)
                # t_aper is in one-way seconds; t1 is the one-way time axis
                tnumber = t_aper / dt1                    # fractional 0-based index
                it0 = np.floor(tnumber).astype(int)       # lower sample (0-based)
                it_next = it0 + 1
                xt0 = tnumber - it0                       # weight for next sample
                xt1 = 1.0 - xt0                           # weight for current sample
                # clamp to valid range
                it0     = np.clip(it0,     0, len(tmp0) - 1)
                it_next = np.clip(it_next, 0, len(tmp0) - 1)
                tmp = xt1 * tmp0[it0] + xt0 * tmp0[it_next]

            elif interp_type == 2:
                f = interp1d(t1, tmp0, kind='cubic',
                             bounds_error=False, fill_value=(tmp0[0], tmp0[-1]))
                tmp = f(t_aper)

            elif interp_type == 3:
                f = interp1d(t1, tmp0, kind='linear',
                             bounds_error=False, fill_value=(tmp0[0], tmp0[-1]))
                tmp = f(t_aper)

            else:  # interp_type == 4
                tmp = sinci(tmp0, t1, t_aper)

            # ---- aperture taper ----
            ccoef = 1.0
            offset_traces = abs(kaper_tr - ktr)
            if offset_traces * dx_val > 0.5 * aper:
                # index into coef1 (0-based): MATLAB uses round(|offset| - traper0)
                # coef1 has length round(traper1)+1, index 0 = at the aperture edge
                idx_taper = int(round(offset_traces - traper0))
                idx_taper = np.clip(idx_taper, 0, len(coef1) - 1)
                ccoef = coef1[idx_taper]
            if abs(1.0 - ccoef) > 0.05:
                tmp = tmp * ccoef

            # ---- obliquity (costheta^1.5) amplitude correction ----
            ind_obl = np.where(costheta < 0.999)[0]
            if len(ind_obl) > 0:
                costheta[ind_obl] = np.sqrt(costheta[ind_obl] ** 3)
                tmp[ind_obl] *= costheta[ind_obl]

            arymig[:, kmig] += tmp

        # ----------------------------------------------------------
        # scaling + 45° phase-shift factor (applied per output trace)
        # ----------------------------------------------------------
        scalemig = aryvel[samptarget, kmig] * np.sqrt(np.pi * (tmig + 0.0001))
        arymig[:, kmig] /= scalemig

    elapsed = time.time() - clock_start
    print(f"migration completed in {int(elapsed)} seconds")

    return arymig, tmig, xmig
