"""
sinci.py  –  8-point sinc function interpolation.

Ports the CREWES MATLAB function sinci.m to Python/NumPy.
Uses Dave Hale's least-squares optimised sinc table approach.
"""

import numpy as np

# Module-level sinc table cache (equivalent to MATLAB's global SINC_TABLE)
_SINC_TABLE: np.ndarray = np.empty((0, 0))
_SINC_TABLE_PARAMS: tuple[int, int] = (0, 0)


def _sinque(x: np.ndarray) -> np.ndarray:
    """Sinc function: sin(pi*x) / (pi*x), with sinque(0) = 1."""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    mask = x != 0.0
    px = np.pi * x[mask]
    out[mask] = np.sin(px) / px
    return out


def _build_sinc_table(lsinc: int, ntable: int) -> np.ndarray:
    """Build the Dave-Hale optimised sinc interpolation table."""
    frac = np.arange(ntable) / ntable
    table = np.zeros((lsinc, ntable))
    jmax = ntable // 2 + 1

    for j in range(jmax):
        fmax = min(0.066 + 0.265 * np.log(lsinc), 1.0)
        a = _sinque(fmax * np.arange(lsinc))
        b = fmax * (np.arange(lsinc // 2 - 1, -lsinc // 2 - 1, -1) + frac[j])
        c = _sinque(b)
        # Toeplitz least-squares design
        from scipy.linalg import toeplitz, lstsq
        T = toeplitz(a)
        table[:, j], _, _, _ = lstsq(T, c)

    # Fill the second half by symmetry
    jtable = ntable - 1
    ktable = 1
    point = lsinc // 2 - 1        # centre index (0-based)
    while jtable > jmax - 1 and table[point, jtable] == 0.0:
        table[:, jtable] = table[::-1, ktable]
        jtable -= 1
        ktable += 1

    return table


def sinci(
    trin: np.ndarray,
    t: np.ndarray,
    tout: np.ndarray,
    sizetable: tuple[int, int] = (8, 25),
) -> np.ndarray:
    """
    8-point sinc function interpolation.

    Parameters
    ----------
    trin : np.ndarray, 1D
        Input (regularly sampled) trace.
    t : np.ndarray, 1D
        Time coordinate vector for trin.  Only the first two values are
        used to determine the sample interval.
    tout : np.ndarray, 1D
        Times at which interpolated amplitudes are desired.
    sizetable : (int, int), optional
        (npts, nfuncs) for the sinc table.  Default = (8, 25).

    Returns
    -------
    trout : np.ndarray, 1D
        Interpolated amplitudes at the times in *tout*.
    """
    global _SINC_TABLE, _SINC_TABLE_PARAMS

    lsinc, ntable = sizetable

    # Rebuild table only if size changed
    if _SINC_TABLE_PARAMS != (lsinc, ntable) or _SINC_TABLE.size == 0:
        _SINC_TABLE = _build_sinc_table(lsinc, ntable)
        _SINC_TABLE_PARAMS = (lsinc, ntable)

    trin = np.asarray(trin, dtype=float).ravel()
    t    = np.asarray(t,    dtype=float).ravel()
    tout = np.asarray(tout, dtype=float).ravel()

    trout = np.zeros(len(tout))
    dtin = t[1] - t[0]

    # Constant extrapolation at boundaries
    mask_lo = tout <= t[0]
    mask_hi = tout >= t[-1]
    trout[mask_lo] = trin[0]
    trout[mask_hi] = trin[-1]

    # Intermediate samples
    ii = np.where((tout > t[0]) & (tout < t[-1]))[0]
    if len(ii) == 0:
        return trout

    pdata = (tout[ii] - t[0]) / dtin         # fractional sample index
    del_ = pdata - np.floor(pdata)            # fractional part in [0, 1)

    # Row index in sinc table
    ptable = (1 + np.round(ntable * del_)).astype(int)

    # Pointer into (padded) trin
    pdata_int = np.floor(pdata).astype(int) + lsinc // 2 - 1

    # Pad trin with edge values
    pad_lo = lsinc // 2 - 1
    pad_hi = lsinc // 2
    trin_pad = np.concatenate([
        np.full(pad_lo, trin[0]),
        trin,
        np.full(pad_hi, trin[-1]),
    ])

    # Fix table overflow
    overflow = ptable == ntable + 1
    ptable[overflow]   = 0
    pdata_int[overflow] += 1

    # Clamp to valid range
    ptable    = np.clip(ptable, 0, ntable - 1)
    max_start = len(trin_pad) - lsinc
    pdata_int = np.clip(pdata_int - lsinc // 2 + 1, 0, max_start)

    # Interpolate via dot product
    for k in range(len(ii)):
        start = pdata_int[k]
        seg   = trin_pad[start: start + lsinc]
        if len(seg) == lsinc:
            trout[ii[k]] = seg @ _SINC_TABLE[:, ptable[k]]

    return trout
