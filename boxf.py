import numpy as np
from scipy.signal import boxcar


def boxf(fwidth, f):
    """
    BOXF: return a boxcar window of width *fwidth* Hz.

    window = boxf(fwidth, f)

    Parameters
    ----------
    fwidth : float
        Width of the desired boxcar in Hz.
    f : array_like
        Frequency coordinate vector (must have at least two samples).

    Returns
    -------
    window : np.ndarray
        Normalised boxcar window (sums to 1 if all weights equal).
    """
    f = np.asarray(f, dtype=float)
    df = f[1] - f[0]
    n = round(fwidth / df)
    if n < 1:
        n = 1
    win = boxcar(n).astype(float) / n
    return win
