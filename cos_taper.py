import numpy as np


def cos_taper(sp: float, ep: float, samp: float = 1.0) -> np.ndarray:
    """
    Compute cosine taper coefficients from sp to ep.

    The taper starts at 1.0 (at sp) and ramps smoothly to 0.0 (at ep)
    following a quarter-cosine curve.

    Parameters
    ----------
    sp : float
        Start point of the taper.
    ep : float
        End point of the taper.
    samp : float, optional
        Sample interval. Default is 1.0.

    Returns
    -------
    coef : np.ndarray
        Taper coefficients of length ``round(|ep - sp| / samp) + 1``.
        Values run from 1.0 down to ~0.0 along a quarter-cosine.
    """
    if samp < 0:
        samp = -samp

    length = round(abs(ep - sp) / samp) + 1

    if length <= 1:
        return np.array([1.0])

    dd = (np.pi / 2.0) / (length - 1)
    coef = np.cos(np.arange(length) * dd)
    return coef
