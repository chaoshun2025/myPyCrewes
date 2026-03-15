import numpy as np
from scipy.signal import hilbert


def phsrot(trin: np.ndarray, theta: float) -> np.ndarray:
    """
    PHSROT – Constant-phase rotation of a trace or section.

    Rotates trin through an angle of theta degrees using the analytic
    (Hilbert-transform) representation:

        trout = cos(θ) * trin  -  sin(θ) * H{trin}

    Parameters
    ----------
    trin : np.ndarray
        Input trace (1D) or section (2D, shape (nsamps, ntraces)).
    theta : float
        Phase rotation angle in degrees.

    Returns
    -------
    trout : np.ndarray
        Phase-rotated output with the same shape as trin.
    """
    trin = np.asarray(trin, dtype=float)
    cos_t = np.cos(np.deg2rad(theta))
    sin_t = np.sin(np.deg2rad(theta))

    if trin.ndim == 1 or min(trin.shape) == 1:
        # Single trace
        trin_1d = trin.ravel()
        trinh = np.imag(hilbert(trin_1d))
        trout = cos_t * trin_1d - sin_t * trinh
        return trout.reshape(trin.shape)

    # Multi-trace section (nsamps, ntraces)
    nsamps, ntraces = trin.shape
    trout = np.zeros_like(trin)
    for k in range(ntraces):
        tmp  = trin[:, k]
        tmph = np.imag(hilbert(tmp))
        trout[:, k] = cos_t * tmp - sin_t * tmph
    return trout
