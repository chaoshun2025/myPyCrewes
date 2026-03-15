"""sincinan.py – 8-point sinc interpolation for signals with embedded NaNs.

Direct port of Margrave toolbox ``sincinan.m`` / ``sinci.m``.
NaN regions in the input do not bleed into adjacent live regions.
"""

import numpy as np
from sinque import sinque

# Module-level cache for the sinc table (mirrors MATLAB's global SINC_TABLE)
_SINC_TABLE = np.zeros((0, 0))
_SINC_TABLE_SIZE = (0, 0)


def _make_sinc_table(lsinc, ntable):
    """Build the lsinc × ntable optimised sinc-coefficient table."""
    frac = np.arange(ntable) / ntable
    table = np.zeros((lsinc, ntable))
    jmax = ntable // 2 + 1

    for j in range(jmax):
        fmax = min(0.066 + 0.265 * np.log(lsinc), 1.0)
        a = sinque(fmax * np.arange(lsinc))
        b = fmax * (np.arange(lsinc // 2 - 1, -lsinc // 2 - 1, -1)
                    + frac[j] * np.ones(lsinc))
        c = sinque(b)
        # Toeplitz least-squares solve
        T = np.zeros((lsinc, lsinc))
        for row in range(lsinc):
            for col in range(lsinc):
                idx = abs(row - col)
                T[row, col] = a[idx] if idx < lsinc else 0.0
        table[:, j] = np.linalg.solve(T, c)

    # Fill the second half by symmetry
    point = lsinc // 2
    jtable = ntable - 1   # 0-based last column
    ktable = 1             # 0-based second column
    while jtable >= 0 and table[point, jtable] == 0.0:
        table[:, jtable] = table[::-1, ktable]
        jtable -= 1
        ktable += 1

    return table


def _get_sinc_table(lsinc=8, ntable=25):
    global _SINC_TABLE, _SINC_TABLE_SIZE
    if _SINC_TABLE.shape != (lsinc, ntable):
        _SINC_TABLE = _make_sinc_table(lsinc, ntable)
        _SINC_TABLE_SIZE = (lsinc, ntable)
    return _SINC_TABLE


def _between(a, b, v):
    """Return indices into *v* where a <= v[i] <= b."""
    v = np.asarray(v)
    return np.where((v >= a) & (v <= b))[0]


def sincinan(trin, t, tout, sizetable=(8, 25)):
    """
    8-point sinc function interpolation with NaN-aware zone handling.

    Parameters
    ----------
    trin : array-like, 1-D
        Input trace (may contain NaNs).
    t : array-like, 1-D
        Uniformly sampled time coordinate for *trin*.
    tout : array-like, 1-D
        Output times at which interpolated values are desired.
    sizetable : tuple (lsinc, ntable), optional
        Shape of the sinc coefficient table.  Default (8, 25).

    Returns
    -------
    trout : np.ndarray, 1-D
        Interpolated trace, same length as *tout*.
        Sites outside the live zones of *trin* are set to NaN.
    """
    trin = np.asarray(trin, dtype=float)
    t = np.asarray(t, dtype=float)
    tout = np.asarray(tout, dtype=float)

    # Handle column-vector input
    trflag = False
    if trin.ndim > 1 and trin.shape[1] == 1:
        trin = trin.ravel()
        t = t.ravel()
        tout = tout.ravel()
        trflag = True

    lsinc, ntable = sizetable
    SINC_TABLE = _get_sinc_table(lsinc, ntable)

    # Find live and dead zones
    ilive = np.where(~np.isnan(trin))[0]
    if len(ilive) == 0:
        return np.full(len(tout), np.nan)

    breaks = np.where(np.diff(ilive) > 1)[0]
    zone_beg = np.concatenate([[ilive[0]],  ilive[breaks + 1]])
    zone_end = np.concatenate([ilive[breaks], [ilive[-1]]])
    nzones = len(zone_beg)

    dtin = t[1] - t[0]
    trout = np.full(len(tout), np.nan)

    for k in range(nzones):
        n1 = int(round((t[zone_beg[k]] - t[0]) / dtin))
        n2 = int(round((t[zone_end[k]] - t[0]) / dtin))
        trinzone = trin[n1:n2 + 1].copy()
        tzone = t[n1:n2 + 1]

        # Output sites inside this zone
        ii = _between(tzone[0], tzone[-1], tout)
        if len(ii) == 0:
            continue

        pdata = (tout[ii] - tzone[0]) / dtin   # 0-based fractional sample
        del_ = pdata - np.floor(pdata)

        # Row index in sinc table (1-based in MATLAB → 0-based here)
        ptable = np.round(ntable * del_).astype(int)
        ptable = np.clip(ptable, 0, ntable - 1)

        # Pointer into (padded) trinzone
        pdata_int = np.floor(pdata).astype(int) + lsinc // 2 - 1

        # Pad trinzone with end values
        half = lsinc // 2
        trinzone_pad = np.concatenate([
            trinzone[0] * np.ones(half - 1),
            trinzone,
            trinzone[-1] * np.ones(half)
        ])

        # Handle edge: if ptable wraps to ntable, reset to 0 and advance pdata
        wrap = ptable == ntable
        ptable[wrap] = 0
        pdata_int[wrap] += 1

        troutzone = np.empty(len(ii))
        for idx in range(len(ii)):
            p = pdata_int[idx]
            seg = trinzone_pad[p - half + 1 : p + half + 1]
            troutzone[idx] = seg @ SINC_TABLE[:, ptable[idx]]

        trout[ii] = troutzone

    if trflag:
        trout = trout[:, np.newaxis]

    return trout
