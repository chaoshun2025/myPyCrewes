import numpy as np


def clip(trin: np.ndarray, amp: float) -> np.ndarray:
    """
    CLIP adjusts only those samples on trin which are greater
    in absolute value than 'amp'. These are set equal to amp but
    with the sign of the original sample.

    Parameters
    ----------
    trin : np.ndarray
        Input trace (1D array).
    amp : float
        Clipping amplitude.

    Returns
    -------
    trout : np.ndarray
        Output trace with samples clipped to [-amp, +amp].
    """
    trout = trin.copy()
    indices = np.abs(trin) > amp
    trout[indices] = np.sign(trin[indices]) * amp
    return trout
