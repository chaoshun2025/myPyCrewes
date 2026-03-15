"""resamp.py – resample a signal using sinc function interpolation.

Direct port of Margrave toolbox ``resamp.m``.  If the output sample
interval is larger than the input (decimation), an anti-alias filter is
applied before the sinc interpolation step.
"""

import numpy as np
from sincinan import sincinan
from filtf import filtf
from trend import trend


def resamp(trin, t, dtout, timesout=None, flag=1, fparms=(0.6, 120.0)):
    """
    Resample a trace using sinc function interpolation.

    Parameters
    ----------
    trin : array-like, 1-D
        Input trace.
    t : array-like, 1-D
        Time coordinate vector for *trin* (uniformly sampled).
    dtout : float
        Desired output sample interval (seconds).
    timesout : sequence of two floats or None, optional
        [tmin, tmax] for the output trace.  The first output sample is
        at tmin; samples continue at dtout spacing until just before tmax.
        Default: [t[0], t[-1]].
    flag : {0, 1}, optional
        Anti-alias filter phase:
        0 → zero-phase,  1 → minimum-phase (default).
    fparms : sequence of two floats, optional
        (f3db, atten_nyq) where
        f3db  = 3-dB corner as a fraction of the *new* Nyquist (default 0.6),
        atten = attenuation desired at the new Nyquist in dB (default 120).

    Returns
    -------
    trout : np.ndarray, 1-D
        Resampled trace.
    tout : np.ndarray, 1-D
        Output time coordinate vector.
    """
    trin = np.asarray(trin, dtype=float)
    t = np.asarray(t, dtype=float)
    fparms = tuple(fparms)

    # Column-vector input → keep a flag so we can restore orientation
    trflag = False
    if trin.ndim > 1 and trin.shape[0] > 1:
        trin = trin.ravel()
        t = t.ravel()
        trflag = True

    dtin = t[1] - t[0]

    if timesout is None:
        timesout = [t[0], t[-1]]

    # Build output time vector
    tout = np.arange(timesout[0], timesout[1] + 1e-10 * dtout, dtout)

    # Find live (non-NaN) samples
    ilive = np.where(~np.isnan(trin))[0]

    if len(ilive) == 0:
        trout = np.full(len(tout), np.nan)
        return (trout[:, np.newaxis] if trflag else trout), tout

    # --- anti-alias filter (only when decimating) ---
    if dtout > dtin:
        fnyq_new = 0.5 / dtout
        fmax_filt = [fparms[0] * fnyq_new,
                     fnyq_new * (1.0 - fparms[0]) * 0.3]

        # Identify contiguous live zones
        breaks = np.where(np.diff(ilive) > 1)[0]
        zone_beg = np.concatenate([[ilive[0]], ilive[breaks + 1]])
        zone_end = np.concatenate([ilive[breaks], [ilive[-1]]])

        for k in range(len(zone_beg)):
            n1 = int(round((t[zone_beg[k]] - t[0]) / dtin))
            n2 = int(round((t[zone_end[k]] - t[0]) / dtin))
            trinzone = trin[n1:n2 + 1].copy()
            tzone = t[n1:n2 + 1]

            # Remove trend, filter, restore trend
            tr, _ = trend(trinzone, tzone, 3)
            trinzone_filt = filtf(
                trinzone - tr, tzone,
                [0.0, 0.0],          # no low-cut
                fmax_filt,
                flag,
                fparms[1]
            )
            trin[n1:n2 + 1] = trinzone_filt + tr

    # --- sinc interpolation ---
    trout = sincinan(trin, t, tout)

    if trflag:
        trout = trout[:, np.newaxis]
        tout = tout[:, np.newaxis]

    return trout, tout
