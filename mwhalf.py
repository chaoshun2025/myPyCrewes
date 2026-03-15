"""mwhalf.py – N-point half-Margrave window (boxcar + trailing taper)."""

import numpy as np


def mwhalf(n, percent=10):
    """
    Return the N-point half-Margrave window as a 1-D NumPy array.

    The window is a boxcar over the first (100 - percent)% of samples,
    with a raised-cosine (Hanning-style) taper on the trailing end only.

    Parameters
    ----------
    n : int or array-like
        Window length.  If an array is supplied, len(n) is used.
    percent : float, optional
        Percentage taper at the trailing end.  Default is 10.

    Returns
    -------
    w : np.ndarray, shape (n,)

    Raises
    ------
    ValueError
        If percent is outside [0, 50].
    """
    if np.ndim(n) > 0:
        n = len(n)
    n = int(n)

    if percent > 50 or percent < 0:
        raise ValueError("invalid percent for mwhalf")

    # m is always even (mirrors MATLAB's 2*floor(m/2))
    m = int(2 * np.floor(percent * n / 100.0))

    if m > 0:
        k = np.arange(m)
        # periodic Hanning (matches MATLAB's hanning(m))
        h = 0.5 - 0.5 * np.cos(2.0 * np.pi * k / m)
        half = m // 2
        # MATLAB: h(m/2:-1:1) → descending half  [index m/2-1 down to 0]
        ramp_down = h[half - 1 :: -1]
    else:
        ramp_down = np.array([])

    plateau = np.ones(n - m // 2)
    w = np.concatenate([plateau, ramp_down])
    return w
