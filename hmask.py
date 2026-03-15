import numpy as np
from scipy.signal import hilbert
from findex import findex


def hmask(trin, thresh=0.05):
    """
    HMASK: compute the Hilbert Reflectivity Mask of a trace.

    mask, htrin = hmask(trin, thresh)

    Returns a trace that is zero everywhere except at the samples
    corresponding to peaks of ``abs(hilbert(trin))``, where it is 1.0.

    Parameters
    ----------
    trin : array_like
        Input trace (1-D).
    thresh : float, optional
        Significance threshold for envelope peaks.  Peaks smaller than
        ``thresh * max_peak`` are suppressed.  Default = 0.05.

    Returns
    -------
    mask : np.ndarray, shape (len(trin),)
        Hilbert mask: 1 at significant envelope peaks, 0 elsewhere.
    htrin : np.ndarray, shape (len(trin),), complex
        Complex analytic signal (i.e. ``scipy.signal.hilbert(trin)``).
    """
    trin = np.asarray(trin, dtype=float).ravel()

    htrin = hilbert(trin)       # complex analytic signal
    env = np.abs(htrin)

    iex = findex(env)           # indices of local envelope maxima

    mask = np.zeros(len(env))
    if len(iex) > 0:
        mask[iex] = 1.0

    # suppress peaks below threshold
    mm = np.max(env)
    iz = np.where(env < mm * thresh)[0]
    mask[iz] = 0.0

    return mask, htrin
