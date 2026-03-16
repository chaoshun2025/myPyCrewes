"""
kirk_migz.py – Post-stack Kirchhoff depth migration.

Faithful Python translation of ``kirk_migz.m`` from the CREWES MATLAB toolbox
(Xinxiang Li / G.F. Margrave, 1996).

Dependencies (must be importable from the same directory):
    near        (near.py)
    cos_taper   (cos_taper.py)
    sinci       (sinci.py)
    interpbl    (interpbl.py)
"""

import time
import numpy as np
from near import near
from cos_taper import cos_taper


# ---------------------------------------------------------------------------
# Private helper: linear taper (mirrors lin_taper.m)
# ---------------------------------------------------------------------------

def _lin_taper(sp: float, ep: float) -> np.ndarray:
    """Linear taper from 1 (at sp) to 0 (at ep)."""
    n = max(int(round(abs(ep - sp))) + 1, 2)
    return np.linspace(1.0, 0.0, n)


# ---------------------------------------------------------------------------
# Private helper: MATLAB-style round (half away from zero)
# ---------------------------------------------------------------------------

def _mround(x: float) -> int:
    """Round half away from zero, matching MATLAB round()."""
    return int(np.floor(x + 0.5))


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def kirk_migz(seis, vmodel, dt, dx, dz, params=None):
    """
    Post-stack Kirchhoff depth migration.

    Parameters
    ----------
    seis : array_like, shape (nsamp, ntr)
        Zero-offset data matrix.  One trace per column.
    vmodel : scalar, 1-D or 2-D array_like
        Velocity model (m/s).  Accepted shapes:

        * scalar → constant velocity everywhere.
        * 1-D of length ``nsamp`` → same velocity profile for every trace.
        * 1-D of length ``ntr``   → one velocity per trace, constant in time.
        * 2-D of shape ``(nsamp, ntr)`` → full velocity matrix.

    dt : scalar or array_like
        If scalar: time sample interval (s).
        If vector of length ``nsamp``: time coordinate vector (s).
    dx : scalar or array_like
        If scalar: spatial sample interval (same units as velocity).
        If vector of length ``ntr``: x coordinate vector.
    dz : scalar or array_like
        Accepted but **not used** in the migration computation (matching the
        MATLAB original).  The output depth axis is taken from the input time
        axis ``t`` via the target-window selection.
    params : array_like of length ≤ 12, optional
        Migration control parameters.  Use ``np.nan`` (or omit trailing
        entries) to accept the default for any parameter.

        ======  ===  ==================================================
        Index   Dim  Description
        ======  ===  ==================================================
        0       [m]  Aperture half-width; default = full section width.
        1       [m]  Aperture taper width; default = aperture / 20.
        2            Aperture taper type: 0 = linear, 1 = cosine (def).
        3       [°]  Max dip angle; default = 60°.
        4       [°]  Angle-limit taper width; default = 15 % of params[3].
        5            Angle taper type: 0 = linear, 1 = cosine (default).
        6            Interpolation: 1=linear (def), 2=spline, 3=cubic,
                     4=sinc.
        7       [s]  ``zmig1`` – output window start time; default=min(t).
        8       [s]  ``zmig2`` – output window end time;   default=max(t).
        9       [m]  ``xmig1`` – output x start;  default = min(x).
        10      [m]  ``xmig2`` – output x end;    default = max(x).
        11           Box-car anti-alias filter: 0 = off (default), 1 = on.
        ======  ===  ==================================================

    Returns
    -------
    seismig : np.ndarray, shape (n_zmig, n_xmig)
        Migrated section.
    zmig : np.ndarray
        Time coordinates of the migrated rows (same as the input time axis
        within the target window – the MATLAB original also returns time,
        not depth, despite the variable name ``zmig``).
    xmig : np.ndarray
        X coordinates of the migrated traces.

    Notes
    -----
    The migration formula is the standard hyperbolic Kirchhoff summation in
    the time domain::

        t_aper = sqrt( offset² / v(z,x)² + (t/2)² )

    which computes the one-way diffraction traveltime for each output
    sample.  An obliquity correction ``cos(θ)^(3/2)`` and a geometric
    scaling ``v * sqrt(π * (t + ε))`` are applied.
    """
    seis    = np.asarray(seis, dtype=float)
    nsamp, ntr = seis.shape

    # ------------------------------------------------------------------
    # Time axis
    # ------------------------------------------------------------------
    dt_in = np.atleast_1d(np.asarray(dt, dtype=float)).ravel()
    if len(dt_in) > 1:
        if len(dt_in) != nsamp:
            raise ValueError("Length of t vector must equal number of rows in seis.")
        t    = dt_in.copy()
        dt_val = float(t[1] - t[0])
    else:
        dt_val = float(dt_in[0])
        t = np.arange(nsamp, dtype=float) * dt_val

    # ------------------------------------------------------------------
    # Spatial axis
    # ------------------------------------------------------------------
    dx_in = np.atleast_1d(np.asarray(dx, dtype=float)).ravel()
    if len(dx_in) > 1:
        if len(dx_in) != ntr:
            raise ValueError("Length of x vector must equal number of columns in seis.")
        x      = dx_in.copy()
        dx_val = float(x[1] - x[0])
    else:
        dx_val = float(dx_in[0])
        x = np.arange(ntr, dtype=float) * dx_val

    # ------------------------------------------------------------------
    # Velocity model → (nsamp, ntr) matrix
    # ------------------------------------------------------------------
    vmodel = np.asarray(vmodel, dtype=float)
    if vmodel.ndim == 0 or vmodel.size == 1:
        vel_mat = float(vmodel) * np.ones((nsamp, ntr))
    elif vmodel.ndim == 1:
        nvsamp = vmodel.size
        if nvsamp == nsamp:
            vel_mat = vmodel[:, np.newaxis] * np.ones((1, ntr))
        elif nvsamp == ntr:
            # transposed vector case (matches MATLAB nvtr==nsamp branch)
            vel_mat = vmodel[np.newaxis, :] * np.ones((nsamp, 1))
        else:
            raise ValueError("Velocity vector is wrong size.")
    else:
        nvsamp, nvtr = vmodel.shape
        if nvsamp == 1 and nvtr == 1:
            vel_mat = float(vmodel[0, 0]) * np.ones((nsamp, ntr))
        elif nvsamp == 1 or nvtr == 1:
            v1 = vmodel.ravel()
            if nvtr == nsamp:            # transposed
                vmodel = vmodel.T
                nvsamp, nvtr = vmodel.shape
                vel_mat = vmodel * np.ones((1, ntr))
            elif nvsamp == nsamp:
                vel_mat = vmodel * np.ones((1, ntr))
            else:
                raise ValueError("Velocity vector is wrong size.")
        else:
            if nvsamp != nsamp:
                raise ValueError("Velocity matrix has wrong number of rows.")
            if nvtr != ntr:
                raise ValueError("Velocity matrix has wrong number of columns.")
            vel_mat = vmodel

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    NPARAMS = 12
    if params is None:
        p = [np.nan] * NPARAMS
    else:
        p = list(np.asarray(params, dtype=float).ravel())
    while len(p) < NPARAMS:
        p.append(np.nan)

    def _p(i, default):
        v = p[i]
        return default if np.isnan(float(v)) else float(v)

    aper        = _p(0,  abs(float(x[-1]) - float(x[0])))
    width1      = _p(1,  aper / 20.0)
    itaper1     = int(_p(2,  1.0))
    ang_limit   = _p(3,  60.0) * (np.pi / 180.0)   # → radians
    # MATLAB: default width2 = 0.15*ang_limit (radians);
    #         if params(5) given: width2 = params(5)*pi/180
    width2_deg  = _p(4,  np.nan)
    if np.isnan(p[4]):
        width2 = 0.15 * ang_limit                   # already in radians
    else:
        width2 = float(p[4]) * (np.pi / 180.0)
    angle1      = ang_limit + width2                 # outer angle limit

    itaper2     = int(_p(5,  1.0))
    if itaper2 not in (0, 1):
        raise ValueError("params[5] (angle taper type) must be 0 or 1.")

    interp_type = int(_p(6,  1.0))
    if not 1 <= interp_type <= 4:
        raise ValueError("params[6] (interpolation type) must be 1, 2, 3, or 4.")

    tmig1       = _p(7,  float(t.min()))     # output window start (time)
    tmig2       = _p(8,  float(t.max()))     # output window end   (time)
    if tmig2 < tmig1:
        raise ValueError("params[8] (zmig2) must be >= params[7] (zmig1).")

    xmig1       = _p(9,  float(x.min()))
    xmig2       = _p(10, float(x.max()))
    if xmig2 < xmig1:
        raise ValueError("params[10] (xmig2) must be >= params[9] (xmig1).")

    ibcfilter   = int(_p(11, 0.0))

    # Cumulative sum for boxcar filter
    if ibcfilter:
        arycum = np.cumsum(seis, axis=0)

    # ------------------------------------------------------------------
    # Aperture: taper coefficient array
    # ------------------------------------------------------------------
    traper0 = 0.5 * aper / dx_val           # half-aperture in trace units
    traper1 = width1 / dx_val               # taper width in trace units
    traper  = _mround(traper0 + traper1)    # total aperture half-width in traces

    if itaper1 == 0:
        coef1 = _lin_taper(traper0, traper0 + traper1)
    else:
        coef1 = cos_taper(traper0, traper0 + traper1)

    # ------------------------------------------------------------------
    # One-way time axis
    # ------------------------------------------------------------------
    dt1 = 0.5 * dt_val          # half-sample interval
    t1  = t / 2.0               # one-way times (half of two-way)
    t2  = t1 ** 2               # squared one-way times (used in diffraction)

    # ------------------------------------------------------------------
    # Maximum traveltime needed → pad seis
    # ------------------------------------------------------------------
    vmin = vel_mat.min()
    tmax = np.sqrt(0.25 * tmig2 ** 2 + ((0.5 * aper + width1) / vmin) ** 2)

    # MATLAB: npad = ceil(tmax/dt1) - nsamp + 5
    npad = int(np.ceil(tmax / dt1)) - nsamp + 5
    if npad > 0:
        seis_pad = np.vstack([seis, np.zeros((npad, ntr))])
        # MATLAB: t1 = [t1', (nsamp+1:nsamp+npad)*dt1]'
        # 1-based: nsamp+1, nsamp+2, ..., nsamp+npad → 0-based equivalent:
        t1_pad = np.concatenate([
            t1,
            np.arange(nsamp + 1, nsamp + npad + 1, dtype=float) * dt1
        ])
        if ibcfilter:
            # Extend arycum by repeating the last row npad times
            arycum = np.vstack([
                arycum,
                np.tile(arycum[-1:, :], (npad, 1))
            ])
    else:
        seis_pad = seis
        t1_pad   = t1

    lentr = nsamp + max(npad, 0)    # total padded length

    # ------------------------------------------------------------------
    # Target windows
    # ------------------------------------------------------------------
    # MATLAB: samptarget = near(t, tmig1, tmig2)  – 1-based indices
    # Python near() returns 0-based indices
    samptarget = near(t, tmig1, tmig2)
    tmig       = t[samptarget]           # time at output rows  (= "zmig")

    trtarget   = near(x, xmig1, xmig2)
    xmig_out   = x[trtarget]

    # ------------------------------------------------------------------
    # Output array
    # ------------------------------------------------------------------
    nz_out = len(samptarget)
    nx_out = len(trtarget)
    seismig = np.zeros((nz_out, nx_out))

    # Precompute quantities at samptarget rows
    t2_target  = t2[samptarget]         # (t/2)^2 at output rows
    tmig_half  = tmig / 2.0             # t/2 at output rows  (= t1 at targets)

    print(f"\n --- Total number of traces to be migrated: {nx_out} ---\n")
    clock1 = time.time()
    kmig   = 0

    for ktr in trtarget:
        kmig += 1

        # Traces within the migration aperture (0-based indices)
        n1    = max(0, ktr - traper)
        n2    = min(ntr - 1, ktr + traper)
        truse = np.arange(n1, n2 + 1)

        # Squared horizontal offsets (one value per aperture trace)
        offset2 = ((truse - ktr) * dx_val) ** 2   # shape (n_aper,)

        # Velocity squared at samptarget depths, current output trace column
        v2_target = vel_mat[samptarget, ktr] ** 2   # shape (nz_out,)

        if kmig % 20 == 0:
            elapsed   = time.time() - clock1
            remaining = (elapsed / kmig) * (int(trtarget[-1]) - ktr)
            print(f" Migrated trace no. {kmig} of {nx_out}, "
                  f"aperture traces: {len(truse)}")
            print(f" Estimated time remaining: {int(remaining)} s")

        # ----------------------------------------------------------
        # Loop over input traces in the aperture
        # ----------------------------------------------------------
        for kaper_idx, kaper_tr in enumerate(truse):

            # ---- diffraction traveltime (one-way) ----------------
            # MATLAB: t_aper = sqrt(offset2(kaper)/v2(samptarget) + t2(samptarget))
            t_aper = np.sqrt(offset2[kaper_idx] / v2_target + t2_target)

            # ---- cosine of dip angle (obliquity correction) ------
            # MATLAB: costheta = 0.5*tmig ./ (t_aper + 100*eps)
            # tmig = t at samptarget; 0.5*tmig = t1 at samptarget
            costheta = tmig_half / (t_aper + 100.0 * np.finfo(float).eps)
            costheta[0] = 1.0   # avoid NaN at t=0

            tanalpha = np.sqrt(np.maximum(1.0 - costheta ** 2, 0.0))

            # ---- angle-limit taper --------------------------------
            # Find last index where costheta < cos(angle1) (outer limit)
            # MATLAB: ind = find(costheta < cos(angle1)); i1 = ind(end)
            ind_outer = np.where(costheta < np.cos(angle1))[0]
            if len(ind_outer) > 0:
                i1 = int(ind_outer[-1])   # 0-based, last beyond outer limit
                # Find last index where costheta < cos(ang_limit) (inner limit)
                ind_inner = np.where(costheta < np.cos(ang_limit))[0]
                if len(ind_inner) > 0:
                    i2 = int(ind_inner[-1])  # 0-based
                    if i1 < i2:
                        # coef2 = cos_taper(i2, i1) in MATLAB 1-based
                        # (sp=i2, ep=i1 → taper from 1 at sp down to 0 at ep)
                        # Note: i1,i2 here are already 0-based Python indices
                        # We need to call with the MATLAB values:
                        # MATLAB i1,i2 are 1-based → MATLAB i2 > MATLAB i1
                        # cos_taper(i2_matlab, i1_matlab) = cos_taper(i2+1, i1+1)
                        # But the LENGTH only depends on |i2-i1|, which is the
                        # same in both 0-based and 1-based. Use 0-based directly.
                        coef2 = cos_taper(float(i2), float(i1))
                        n_tap = i2 - i1   # = len(coef2) - 1

                        # Zero out samples beyond outer limit
                        costheta[: i1 + 1] = 0.0

                        # Taper samples between i1 and i2
                        # MATLAB: coef2(i2-i1:-1:1)' .* costheta(i1+1:i2)
                        # coef2 has n_tap+1 elements (0..n_tap);
                        # select indices n_tap-1, n_tap-2, ..., 0 → reversed
                        taper_vals = coef2[n_tap - 1::-1]  # length n_tap
                        costheta[i1 + 1: i2 + 1] *= taper_vals

            # ---- boxcar anti-alias filter -------------------------
            if ibcfilter:
                # MATLAB: lt0 = round(dx*tanalpha./vmodel(samptarget,ktr)/dt1)
                lt0  = np.round(
                    dx_val * tanalpha / vel_mat[samptarget, ktr] / dt1
                ).astype(int)
                # MATLAB: indt = round(t_aper/dt1) + 1  (1-based)
                indt = (np.round(t_aper / dt1) + 1).astype(int)   # 1-based
                indt_py = indt - 1                                 # 0-based

                lt = np.ones(lentr, dtype=int) * lt0.max()
                indt_clamped = np.clip(indt_py, 0, lentr - 1)
                lt[indt_clamped] = lt0
                # MATLAB: lt(max(indt)+1:lentr) = min(lt0)
                lt[int(indt_py.max()) + 1:] = lt0.min()

                # MATLAB: it = (1:lentr)'  (1-based)
                it   = np.arange(1, lentr + 1)   # 1-based
                l1   = it - lt - 1               # 1-based indices (may be 0 or negative)
                l2   = it + lt                   # 1-based

                # Clamp to [1, lentr] (1-based)
                l1   = np.clip(l1, 1, lentr)
                l2   = np.clip(l2, 1, lentr)

                # Convert to 0-based for numpy indexing
                l1_py = l1 - 1
                l2_py = l2 - 1

                tmp0         = np.empty(lentr)
                tmp0[0]      = arycum[0, kaper_tr]
                dlen         = l2_py[1:] - l1_py[1:]
                with np.errstate(divide="ignore", invalid="ignore"):
                    tmp0[1:] = np.where(
                        dlen > 0,
                        (arycum[l2_py[1:], kaper_tr]
                         - arycum[l1_py[1:], kaper_tr]) / dlen,
                        0.0,
                    )
            else:
                tmp0 = seis_pad[:, kaper_tr]

            # ---- interpolation ------------------------------------
            if interp_type == 1:
                # Linear interpolation – exact MATLAB translation
                # MATLAB: tnumber = t_aper/dt1
                #         it0 = floor(tnumber) + 1  (1-based)
                #         it1 = it0 + 1
                #         xt0 = tnumber - it0 + 1   (fraction in [0,1))
                #         xt1 = it0 - tnumber        (= 1 - xt0)
                tnumber = t_aper / dt1
                it0_py  = np.clip(
                    np.floor(tnumber).astype(int), 0, len(tmp0) - 1
                )
                it1_py  = np.clip(it0_py + 1, 0, len(tmp0) - 1)
                xt0     = tnumber - np.floor(tnumber)   # fraction
                xt1     = 1.0 - xt0
                tmp     = xt1 * tmp0[it0_py] + xt0 * tmp0[it1_py]

            elif interp_type == 2:
                # Spline interpolation
                from scipy.interpolate import interp1d
                f_interp = interp1d(
                    t1_pad, tmp0, kind='spline' if False else 'cubic',
                    bounds_error=False,
                    fill_value=(float(tmp0[0]), float(tmp0[-1]))
                )
                tmp = f_interp(t_aper)

            elif interp_type == 3:
                # Cubic interpolation
                from scipy.interpolate import interp1d
                f_interp = interp1d(
                    t1_pad, tmp0, kind='cubic',
                    bounds_error=False,
                    fill_value=(float(tmp0[0]), float(tmp0[-1]))
                )
                tmp = f_interp(t_aper)

            elif interp_type == 4:
                # Sinc interpolation
                from sinci import sinci
                tmp = sinci(tmp0, t1_pad, t_aper)

            else:
                raise ValueError(f"Unknown interpolation type: {interp_type}")

            # ---- aperture taper ----------------------------------
            # MATLAB: if abs(truse(kaper)-ktr)*dx > 0.5*aper
            #             ccoef = coef1(round(abs(truse(kaper)-ktr)-traper0))
            ccoef = 1.0
            offset_tr = abs(int(kaper_tr) - int(ktr)) * dx_val
            if offset_tr > 0.5 * aper:
                delta    = abs(int(kaper_tr) - int(ktr)) - traper0
                # MATLAB coef1 is 1-indexed; coef1(k) → Python coef1[k-1]
                # MATLAB index = round(delta); Python = max(0, round(delta)-1)
                idx_1based = _mround(delta)
                idx_py     = np.clip(idx_1based - 1, 0, len(coef1) - 1)
                ccoef      = coef1[idx_py]
            if abs(1.0 - ccoef) > 0.05:
                tmp = tmp * ccoef

            # ---- obliquity amplitude weighting -------------------
            # MATLAB: ind = find(costheta < 0.999)
            #         costheta(ind) = sqrt(costheta(ind).^3)
            #         tmp(ind) = tmp(ind) .* costheta(ind)
            ind_apply = np.where(costheta < 0.999)[0]
            costheta[ind_apply] = np.sqrt(costheta[ind_apply] ** 3)
            tmp[ind_apply] *= costheta[ind_apply]

            seismig[:, kmig - 1] += tmp

        # ----------------------------------------------------------
        # Scaling: v * sqrt(pi * (t + 0.0001))
        # MATLAB: scalemig = vmodel(samptarget, kmig) .* sqrt(pi*(tmig+0.0001))
        # Note: kmig is 1-based in MATLAB; it indexes the OUTPUT column.
        # vmodel(samptarget, kmig) uses the kmig-th column of the velocity
        # matrix at the samptarget rows.
        # ----------------------------------------------------------
        scalemig = vel_mat[samptarget, kmig - 1] * np.sqrt(
            np.pi * (tmig + 1e-4)
        )
        seismig[:, kmig - 1] /= scalemig

    elapsed = time.time() - clock1
    print(f"Migration completed in {int(elapsed)} seconds")

    # zmig is the time axis at target rows (MATLAB returns tmig as zmig)
    zmig = tmig
    return seismig, zmig, xmig_out
