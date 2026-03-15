"""hilbm.py – Hilbert transform requiring power-of-2 input length.

Direct port of the Margrave toolbox ``hilbm.m``.
"""

import numpy as np


def hilbm(x):
    """
    Hilbert transform (Margrave toolbox variant).

    The input length **must** be a power of 2; an error is raised otherwise.
    The real part of the input is used; the imaginary part is discarded.

    Parameters
    ----------
    x : array-like (real or complex), 1-D
        Input signal.  len(x) must be a power of 2.

    Returns
    -------
    y : np.ndarray (complex)
        Analytic signal: real(y) = original real data,
        imag(y) = Hilbert transform.

    Raises
    ------
    ValueError
        If len(x) is not a power of 2.
    """
    x = np.asarray(x)
    m = len(x)

    if m & (m - 1):          # not a power of 2
        raise ValueError(
            f"input vector length must be a power of 2, got {m}"
        )

    yy = np.fft.fft(np.real(x))

    if m > 1:
        h = np.zeros(m)
        h[0] = 1.0
        h[1 : m // 2] = 2.0
        h[m // 2] = 1.0      # Nyquist bin — keep as-is (matches MATLAB)
        # MATLAB: h = [1; 2*ones(m/2,1); zeros(m-m/2-1,1)]
        # index 0 → 1, indices 1..m/2-1 → 2, indices m/2..m-1 → 0
        h = np.zeros(m)
        h[0] = 1.0
        h[1 : m // 2] = 2.0
        # indices m//2 onward stay 0 (already zero)
        yy = yy * h

    return np.fft.ifft(yy)
