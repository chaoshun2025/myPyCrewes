"""fftrl.py – forward Fourier transform for real-valued signals.

Returns only the positive-frequency half of the spectrum, matching the
convention used throughout the Margrave seismic toolbox.
"""

import numpy as np
from mwindow import mwindow


def fftrl(s, t, percent=0.0, npad=None):
    """
    Forward Fourier transform of a real-valued signal (positive freqs only).

    If *s* is a 1-D array of length N the output has floor(N/2)+1 samples.
    If *s* is a 2-D array each *column* is transformed independently.

    Parameters
    ----------
    s : array-like, 1-D or 2-D
        Input signal(s).  For a 2-D input the first axis is time.
    t : array-like
        Time coordinate vector.  Only t[0] and t[1]-t[0] (i.e. dt) matter
        for the frequency axis; the length must match s along axis-0 (or
        axis-1 for a single row vector).
    percent : float, optional
        Length of raised-cosine taper applied to both ends before the FFT,
        expressed as a percentage of the signal length.  0 = no taper.
        Default 0.
    npad : int or None, optional
        Total length to which the signal is zero-padded before the FFT.
        None (default) → use len(t) (i.e. no padding beyond the input).
        0 is treated the same as None.

    Returns
    -------
    spec : np.ndarray (complex)
        Positive-frequency spectrum.  Shape is (nf, …) where
        nf = floor(npad/2) + 1.
    f : np.ndarray
        Frequency vector of length nf (Hz).

    Notes
    -----
    * MATLAB's FFTRL uses the **+** sign in the Fourier exponential (i.e.
      ``fft`` convention), so this function does the same via ``np.fft.fft``.
    * The companion inverse is :func:`ifftrl`.
    """
    s = np.asarray(s, dtype=float)

    # --- handle row-vector input (1 × N) ---
    itr = False
    if s.ndim == 1:
        nsamps = len(s)
    elif s.shape[0] == 1:          # row vector
        s = s.ravel()
        nsamps = len(s)
        itr = True
    else:
        nsamps = s.shape[0]

    t = np.asarray(t, dtype=float)
    if len(t) != nsamps:
        # rebuild t from dt only
        dt = t[1] - t[0]
        t = t[0] + dt * np.arange(nsamps)

    if npad is None or npad == 0:
        npad = nsamps

    # --- taper ---
    if percent > 0:
        w = mwindow(nsamps, percent)
        if s.ndim == 1:
            s = s * w
        else:
            s = s * w[:, np.newaxis]

    # --- FFT ---
    spec = np.fft.fft(s, n=npad, axis=0)
    nf = int(npad // 2) + 1
    spec = spec[:nf]

    # --- frequency axis ---
    dt = t[1] - t[0]
    fnyq = 0.5 / dt
    df = 2.0 * fnyq / npad
    f = df * np.arange(nf)

    if itr:
        f = f.ravel()
        spec = spec.ravel()

    return spec, f
