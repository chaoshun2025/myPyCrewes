"""wavenorm.py – normalise a wavelet by one of three criteria."""

import numpy as np
from fftrl import fftrl
from todb import todb


def wavenorm(w, tw, flag):
    """
    Normalise a wavelet.

    Parameters
    ----------
    w : array-like, 1-D
        Input wavelet.
    tw : array-like, 1-D
        Time coordinate vector for *w*.
    flag : {1, 2, 3}
        1 → normalise so that max(|w|) = 1.
        2 → normalise so that a sine wave at the dominant frequency
            passes with unit amplitude (convolution-based, matches MATLAB).
        3 → normalise so that the RMS amplitude = 1.

    Returns
    -------
    wnorm : np.ndarray
        Normalised wavelet, same shape as *w*.

    Raises
    ------
    ValueError
        For an invalid flag value.
    """
    w = np.asarray(w, dtype=float)
    tw = np.asarray(tw, dtype=float)

    if flag == 1:
        return w / np.max(np.abs(w))

    elif flag == 2:
        W, f = fftrl(w, tw)
        A = np.real(todb(W))
        ind = np.argmax(A)
        fdom = f[ind]
        refwave = np.sin(2.0 * np.pi * fdom * tw)
        # convz equivalent: full linear convolution, zero-padded to same size
        reftest = np.convolve(refwave, w, mode='full')
        # MATLAB convz keeps the central part (same length as refwave)
        n = len(refwave)
        start = (len(reftest) - n) // 2
        reftest = reftest[start:start + n]
        fact = np.max(refwave) / np.max(reftest)
        return w * fact

    elif flag == 3:
        rms = np.linalg.norm(w) / np.sqrt(len(w))
        return w / rms

    else:
        raise ValueError(f"invalid flag: {flag!r}")
