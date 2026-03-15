import numpy as np
from between import between


def gaussian(x, xnot, xwid, yht=1.0):
    """
    GAUSSIAN: evaluate a Gaussian function on a coordinate axis.

    y = gaussian(x, xnot, xwid, yht)

    Parameters
    ----------
    x : array_like
        x coordinate axis.
    xnot : float
        Centre of the Gaussian.
    xwid : float
        Width of the Gaussian.  The standard deviation is sigma = xwid / 4.
    yht : float, optional
        Peak height of the Gaussian.  Default = 1.

    Returns
    -------
    y : np.ndarray, shape (len(x),)
        Column vector containing the Gaussian values.

    Notes
    -----
    To window a signal ``s`` with time coordinate ``t``::

        gau = gaussian(t, np.mean(t), (t[-1] - t[0]) / 10)
        sw  = s * gau

    At 2*sigma from xnot the Gaussian is 17.4 dB down;
    at 4*sigma it is 69.5 dB down.
    """
    x = np.asarray(x, dtype=float).ravel()

    if len(between(x[0], x[-1], np.array([xnot]), flag=2)) == 0:
        raise ValueError("xnot not contained in x")

    sigma = xwid / 4.0
    y = yht * np.exp(-0.5 * ((x - xnot) / sigma) ** 2)
    return y
