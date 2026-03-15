"""
kirk_migz.py  –  Post-stack Kirchhoff depth migration.

Ports kirk_migz.m from the CREWES MATLAB toolbox.
Structurally identical to kirk_mig (time migration) except the output
axis is depth (z) rather than time (t).

Depends on:
    cos_taper  (from cos_taper.py)
    sinci      (from sinci.py)
"""

import numpy as np
import time
from cos_taper import cos_taper
from sinci import sinci


# ---------------------------------------------------------------------------
# Helpers (same small utilities as in kirk_mig.py)
# ---------------------------------------------------------------------------

def _lin_taper(start: float, end: float) -> np.ndarray:
    n = max(int(round(abs(end - start))) + 1, 2)
    return np.linspace(1.0, 0.0, n)


def _near(vec: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return np.where((vec >= vmin) & (vec <= vmax))[0]


def _parse_param(params, i, default):
    v = params[i]
    try:
        if np.isnan(float(v)):
            return default
    except (TypeError, ValueError):
        pass
    return v


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def kirk_migz(
    seis: np.ndarray,
    vmodel,
    dt,
    dx,
    dz,
    params=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Post-stack Kirchhoff depth migration.

    Parameters
    ----------
    seis : np.ndarray, shape (nsamp, ntr)
        Zero-offset data matrix (one trace per column).
    vmodel : scalar, 1D array, or 2D array
        Interval velocity model (same spatial dimensions as seis).
    dt : scalar or 1D array
        Time sample interval (s) or time coordinate vector.
    dx : scalar or 1D array
        Spatial sample interval or x coordinate vector.
    dz : scalar or 1D array
        Output depth sample interval, or vector of output depths.
    params : array-like, length up to 12, optional
        Migration parameters (use NaN for defaults); same layout as
        kirk_mig.py params[0..11].

    Returns
    -------
    seismig : np.ndarray, shape (nz_out, nx_out)
        Migrated depth section.
    zmig : np.ndarray
        Depth coordinates of migrated data.
    xmig : np.ndarray
        X coordinates of migrated data.
    """
    seis = np.asarray(seis, dtype=float)
    nsamp, ntr = seis.shape

    # ---- time axis -------------------------------------------------------
    dt_arr = np.atleast_1d(dt)
    if len(dt_arr) > 1:
        if len(dt_arr) != nsamp:
            raise ValueError("Length of t must equal number of rows in seis.")
        t = dt_arr.astype(float).ravel()
        dt_val = t[1] - t[0]
    else:
        dt_val = float(dt)
        t = np.arange(nsamp) * dt_val

    # ---- x axis ----------------------------------------------------------
    dx_arr = np.atleast_1d(dx)
    if len(dx_arr) > 1:
        if len(dx_arr) != ntr:
            raise ValueError("Length of x must equal number of columns in seis.")
        x = dx_arr.astype(float).ravel()
        dx_val = x[1] - x[0]
    else:
        dx_val = float(dx)
        x = np.arange(ntr) * dx_val

    # ---- velocity model --------------------------------------------------
    vmodel = np.asarray(vmodel, dtype=float)
    if vmodel.ndim == 0 or vmodel.size == 1:
        vel_mat = float(vmodel) * np.ones((nsamp, ntr))
    elif vmodel.ndim == 1:
        v = vmodel.ravel()
        if len(v) == nsamp:
            vel_mat = v[:, None] * np.ones((1, ntr))
        elif len(v) == ntr:
            vel_mat = np.ones((nsamp, 1)) * v[None, :]
        else:
            raise ValueError("Velocity vector has wrong size.")
    else:
        if vmodel.shape != (nsamp, ntr):
            raise ValueError("Velocity matrix shape must match seis.")
        vel_mat = vmodel

    # ---- depth axis ------------------------------------------------------
    dz_arr = np.atleast_1d(dz)
    if len(dz_arr) > 1:
        z_out = dz_arr.astype(float).ravel()
        dz_val = z_out[1] - z_out[0]
    else:
        dz_val = float(dz)
        # default depth range: 0 to v_ave_max * Tmax / 2
        vave_max = np.sqrt(np.mean(vel_mat ** 2, axis=0)).max()
        zmax_default = vave_max * t[-1] / 2.0
        z_out = np.arange(0.0, zmax_default + dz_val / 2, dz_val)

    # ---- default migration parameters ------------------------------------
    nparams = 12
    if params is None:
        params = [np.nan] * nparams
    else:
        params = list(params)
    while len(params) < nparams:
        params.append(np.nan)

    def _p(i, default):
        return _parse_param(params, i, default)

    aper        = _p(0,  abs(x[-1] - x[0]))
    width1      = _p(1,  aper / 20.0)
    itaper1     = int(_p(2, 1))
    ang_limit   = _p(3, 60.0) * np.pi / 180.0
    width2      = _p(4, 0.15 * float(_p(3, 60.0))) * np.pi / 180.0
    angle1      = ang_limit + width2
    itaper2     = int(_p(5, 1))
    interp_type = int(_p(6, 1))
    zmig1       = _p(7,  z_out.min())
    zmig2       = _p(8,  z_out.max())
    xmig1       = _p(9,  x.min())
    xmig2       = _p(10, x.max())
    ibcfilter   = int(_p(11, 0))

    # ---- aperture taper --------------------------------------------------
    traper0 = 0.5 * aper / dx_val
    traper1 = width1 / dx_val
    traper  = int(round(traper0 + traper1))
    if itaper1 == 0:
        coef1 = _lin_taper(traper0, traper0 + traper1)
    else:
        coef1 = cos_taper(traper0, traper0 + traper1)

    # ---- one-way time axis -----------------------------------------------
    dt1 = 0.5 * dt_val
    t1  = t / 2.0

    # ---- maximum input time needed ---------------------------------------
    vmin = vel_mat.min()
    tmax_needed = np.sqrt(0.25 * t[-1] ** 2 +
                          ((0.5 * aper + width1) / vmin) ** 2)

    # pad seis in time if necessary
    npad = int(np.ceil(tmax_needed / dt1)) - nsamp + 5
    if npad > 0:
        seis_pad = np.vstack([seis, np.zeros((npad, ntr))])
        t1_pad   = np.concatenate([t1, (nsamp + np.arange(npad)) * dt1])
    else:
        seis_pad = seis
        t1_pad   = t1

    if ibcfilter:
        arycum = np.cumsum(seis_pad, axis=0)

    # ---- target windows --------------------------------------------------
    ztarget  = _near(z_out, zmig1, zmig2)
    zmig     = z_out[ztarget]
    trtarget = _near(x, xmig1, xmig2)
    xmig     = x[trtarget]

    seismig = np.zeros((len(ztarget), len(trtarget)))

    # ---- velocity at output depth levels (average / RMS over time) -------
    # For depth migration the velocity at each depth node is the interval
    # velocity at the two-way time corresponding to that depth and position.
    # Here we use a simple vertical ray (v·t/2 = z) approximation to map
    # depth back to time for amplitude scaling.

    print(f"\n --- Total traces to migrate: {len(trtarget)} ---\n")
    clock1 = time.time()
    kmig = 0

    for ktr in trtarget:
        kmig += 1

        n1 = max(0, ktr - traper)
        n2 = min(ntr - 1, ktr + traper)
        truse = np.arange(n1, n2 + 1)

        offset2 = ((truse - ktr) * dx_val) ** 2

        if kmig % 20 == 0:
            elapsed   = time.time() - clock1
            remaining = (elapsed / kmig) * (trtarget[-1] - ktr)
            print(f" Migrated trace {kmig} of {len(trtarget)}, "
                  f"aperture traces: {len(truse)}")
            print(f"  Estimated time remaining: {remaining:.0f} s")

        for ki, kaper_tr in enumerate(truse):

            # --- travel time for each output depth node -------------------
            # Approximate vertical velocity at the output position/depth
            # v_avg(z) from the time-domain velocity column at ktr.
            # Two-way time corresponding to each output depth:
            #   t_vert(z) = 2*z / v_interval  (approx – use mean velocity)
            # For the full diffraction path:
            #   t_aper = 2 * sqrt( (offset/2)^2 + z^2 ) / v
            # We use the instantaneous velocity at that depth for v.

            # Map output depths to two-way time using the velocity column
            # at the migration trace (simple vertical ray approximation)
            vel_col = vel_mat[:, ktr]           # velocity vs. two-way time
            t_vert  = 2.0 * zmig / np.interp(
                zmig,
                np.cumsum(vel_col * dt_val) - vel_col[0] * dt_val,
                np.arange(nsamp) * dt_val + 1e-10,
            )
            # More robust: use average velocity to convert depth → time
            # v_avg(z) ≈ sqrt( (1/nsamp) * sum(v^2) ) for RMS approximation
            # For simplicity use interpolated instantaneous velocity at depth
            # via a cumulative depth axis
            depth_axis = np.cumsum(vel_col * dt_val)   # depth vs. time sample
            vel_at_z   = np.interp(zmig, depth_axis,
                                   vel_col, left=vel_col[0], right=vel_col[-1])

            # Diffraction traveltime (one-way * 2, converted to half-time)
            t_aper = np.sqrt(offset2[ki] / vel_at_z ** 2 +
                             (zmig / vel_at_z) ** 2)
            # t_aper is now the half-time (i.e. relative to t1 = t/2 axis)

            # Cosine-of-angle correction
            costheta = zmig / (vel_at_z * t_aper + 100 * np.finfo(float).eps)
            costheta = np.clip(costheta, 0.0, 1.0)
            if len(costheta):
                costheta[0] = 1.0

            tanalpha = np.sqrt(np.clip(1.0 - costheta ** 2, 0.0, None))

            # Angle taper
            ind_out = np.where(costheta < np.cos(angle1))[0]
            if len(ind_out):
                i1 = ind_out[-1]
                ind_in = np.where(costheta < np.cos(ang_limit))[0]
                if len(ind_in):
                    i2 = ind_in[-1]
                    if i1 < i2:
                        if itaper2 == 0:
                            coef2 = _lin_taper(i2, i1)
                        else:
                            coef2 = cos_taper(i2, i1)
                        costheta[:i1 + 1] = 0.0
                        n_tap = i2 - i1
                        costheta[i1 + 1: i2 + 1] *= coef2[n_tap - 1::-1][:n_tap]

            # Anti-aliasing boxcar filter
            if ibcfilter:
                lt0  = np.round(dx_val * tanalpha / vel_at_z / dt1).astype(int)
                indt = np.round(t_aper / dt1).astype(int)
                lentr = len(t1_pad)
                lt = np.ones(lentr, dtype=int) * lt0.max()
                lt[np.clip(indt, 0, lentr - 1)] = lt0
                it   = np.arange(lentr)
                l1   = np.clip(it - lt - 1, 0, lentr - 1)
                l2   = np.clip(it + lt,     0, lentr - 1)
                tmp0 = np.empty(lentr)
                tmp0[0] = arycum[0, kaper_tr]
                dlen = l2[1:] - l1[1:]
                tmp0[1:] = np.where(
                    dlen > 0,
                    (arycum[l2[1:], kaper_tr] - arycum[l1[1:], kaper_tr]) / dlen,
                    0.0,
                )
            else:
                tmp0 = seis_pad[:, kaper_tr]

            # Interpolation
            if interp_type == 1:
                tnumber = t_aper / dt1
                it0 = np.clip(np.floor(tnumber).astype(int), 0, len(tmp0) - 1)
                it1 = np.clip(it0 + 1, 0, len(tmp0) - 1)
                xt0 = tnumber - np.floor(tnumber)
                tmp = (1.0 - xt0) * tmp0[it0] + xt0 * tmp0[it1]
            elif interp_type == 2:
                tmp = np.interp(t_aper, t1_pad, tmp0,
                                left=tmp0[0], right=tmp0[-1])
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
                idx   = int(round(abs(kaper_tr - ktr) - traper0))
                idx   = np.clip(idx, 0, len(coef1) - 1)
                ccoef = coef1[idx]
                if abs(1.0 - ccoef) > 0.05:
                    tmp *= ccoef

            # Amplitude weighting
            ind_apply = np.where(costheta < 0.999)[0]
            costheta[ind_apply] = np.sqrt(costheta[ind_apply] ** 3)
            tmp[ind_apply] *= costheta[ind_apply]

            seismig[:, kmig - 1] += tmp

        # Scaling: depth-domain equivalent of the time-domain sqrt(pi*t) factor
        depth_col     = np.cumsum(vel_mat[:, kmig - 1] * dt_val)
        vel_at_zmig   = np.interp(zmig, depth_col,
                                  vel_mat[:, kmig - 1],
                                  left=vel_mat[0, kmig - 1],
                                  right=vel_mat[-1, kmig - 1])
        t_for_scale   = zmig / (vel_at_zmig + 1e-10)
        scalemig      = vel_at_zmig * np.sqrt(np.pi * (t_for_scale + 1e-4))
        seismig[:, kmig - 1] /= scalemig

    elapsed = time.time() - clock1
    print(f"Migration completed in {elapsed:.0f} seconds")

    return seismig, zmig, xmig
