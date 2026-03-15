import numpy as np
from between import between


def triangle(x, xnot, xwid, yht=1.0):
    """
    TRIANGLE: evaluate a triangular window on a coordinate axis.

    y = triangle(x, xnot, xwid, yht)

    Parameters
    ----------
    x : array_like
        x coordinate axis.
    xnot : float
        Centre of the triangle.
    xwid : float
        Full width of the triangle (base-to-base).
    yht : float, optional
        Peak height of the triangle.  Default = 1.

    Returns
    -------
    y : np.ndarray, shape (len(x),)
        Triangle values; zero outside [xnot - xwid/2, xnot + xwid/2].

    Notes
    -----
    To window a signal ``s`` with time coordinate ``t``::

        tri = triangle(t, np.mean(t), (t[-1]-t[0])/10)
        sw  = s * tri

    To truncate to only non-zero samples::

        ind = tri > 0
        swt, twt = sw[ind], t[ind]
    """
    x = np.asarray(x, dtype=float).ravel()

    if len(between(x[0], x[-1], np.array([xnot]), flag=2)) == 0:
        raise ValueError("xnot not contained in x")

    y = np.zeros(len(x))

    # rising flank: xnot - xwid/2  to  xnot
    ind = between(xnot - 0.5 * xwid, xnot, x, flag=2)
    if len(ind) > 0:
        y[ind] = np.linspace(0, yht, len(ind))

    # falling flank: xnot  to  xnot + xwid/2
    ind = between(xnot, xnot + 0.5 * xwid, x, flag=2)
    if len(ind) > 0:
        y[ind] = np.linspace(yht, 0, len(ind))

    return y
