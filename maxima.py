import numpy as np
from scipy.interpolate import CubicSpline


def maxima(trin, mask):
    """
    MAXIMA: spline-refined local maximum amplitudes.

    trout = maxima(trin, mask)

    Takes an input trace and a Hilbert mask (see ``hmask``) and uses
    cubic-spline interpolation to refine the estimates of the local maxima.

    Parameters
    ----------
    trin : array_like
        Input trace (1-D).
    mask : array_like
        Hilbert mask as returned by ``hmask``.  Zero everywhere except at
        samples corresponding to local maxima of the Hilbert envelope of
        trin, where it equals 1.0.

    Returns
    -------
    trout : np.ndarray
        Vector of spline-interpolated maximum amplitudes, one per unit
        spike in mask.
    """
    trin = np.asarray(trin, dtype=float).ravel()
    mask = np.asarray(mask, dtype=float).ravel()

    iex = np.where(mask == 1)[0]
    trout = np.zeros(len(iex))

    for i, idx in enumerate(iex):
        # 5-sample window centred on idx (clipped to trace bounds)
        xs = np.arange(max(0, idx - 2), min(len(trin), idx + 3))
        xi = np.linspace(xs[0], xs[-1], (len(xs) - 1) * 5 + 1)
        cs = CubicSpline(xs, trin[xs])
        trout[i] = np.max(cs(xi))

    return trout
