import numpy as np
from scipy.signal import fftconvolve


def conv45(aryin: np.ndarray) -> np.ndarray:
    """
    CONV45 performs a 45-degree phase shift for the traces of the input
    dataset. The process is performed in the time domain using a fixed
    FIR filter derived from the CREWES MATLAB toolbox (J.C. Bancroft).

    Parameters
    ----------
    aryin : np.ndarray
        Input data. Either a 1D array (single trace) or a 2D array with
        shape (nsamples, ntraces) — one trace per column.

    Returns
    -------
    aryout : np.ndarray
        Phase-shifted output with the same shape as aryin.
    """
    # Fixed 45-degree phase-shift FIR filter coefficients
    filt = np.array([
        -0.0010, -0.0030, -0.0066, -0.0085, -0.0060, -0.0083, -0.0107,
        -0.0164, -0.0103, -0.0194, -0.0221, -0.0705,  0.0395, -0.2161,
        -0.3831,  0.5451,  0.4775, -0.1570,  0.0130,  0.0321, -0.0129
    ])  # length 21; zero-phase centre at index 15 (0-based: 15)

    transposed = False
    aryin = np.atleast_2d(aryin)
    if aryin.shape[0] == 1:          # row vector → treat as column
        aryin = aryin.T
        transposed = True

    nrow, nvol = aryin.shape
    aryout = np.zeros_like(aryin, dtype=float)

    # The MATLAB code selects conv1(16:nrow+15) using 1-based indexing,
    # which corresponds to indices 15:nrow+15 in 0-based Python indexing.
    delay = 15  # samples to skip from the start of the full convolution

    for j in range(nvol):
        conv1 = np.convolve(aryin[:, j].astype(float), filt)
        aryout[:, j] = conv1[delay: delay + nrow]

    if transposed:
        aryout = aryout.T
        return aryout[0] if aryout.shape[0] == 1 else aryout

    # Return 1D if a single column
    if aryout.shape[1] == 1:
        return aryout[:, 0]
    return aryout
