import numpy as np


def burg(trin, t, lc, n=None):
    """
    BURG: Burg (Maximum Entropy) amplitude spectral estimate.

    specb, fb = burg(trin, t, lc, n)

    If trin is a matrix (traces in columns), an ensemble Burg spectrum is
    computed by averaging the prediction filters across traces.

    Parameters
    ----------
    trin : array_like
        Input trace (1-D column preferred) or trace matrix (traces in columns).
    t : array_like
        Time coordinate vector for trin (same number of rows).
    lc : int
        Number of points in the prediction filter (4–20 typical).
    n : int, optional
        Number of points in the final spectral estimate.
        Default = len(trin) (for 1-D) or number of rows (for 2-D).

    Returns
    -------
    specb : np.ndarray
        Burg amplitude spectrum (1-D, length n//2 + 1).
    fb : np.ndarray
        Frequency coordinate vector.

    Notes
    -----
    Requires fftrl to be importable from the same package.
    Adapted from Claerbout (1976) "Fundamentals of Geophysical Data Processing".
    """
    from fftrl import fftrl  # project-local import

    trin = np.asarray(trin, dtype=float)
    t = np.asarray(t, dtype=float).ravel()

    if trin.ndim == 1:
        trin = trin.reshape(-1, 1)

    lx, ntraces = trin.shape

    if n is None:
        n = lx

    sbar = np.zeros(lc)

    for itr in range(ntraces):
        tr = trin[:, itr]
        a = np.zeros(lc)
        a[0] = 1.0
        em = tr.copy()
        ep = tr.copy()

        for j in range(1, lc):
            bot = (np.dot(ep[j:lx], ep[j:lx]) +
                   np.dot(em[: lx - j], em[: lx - j]))
            top = np.dot(ep[j:lx], em[: lx - j])
            cj = 2.0 * top / bot if bot != 0 else 0.0

            epp = np.concatenate([ep[:j],
                                   ep[j:lx] - cj * em[: lx - j]])
            em = np.concatenate([em[: lx - j] - np.conj(cj) * ep[j:lx],
                                  em[lx - j:]])
            ep = epp

            s = a[: j + 1] - cj * np.conj(a[j::-1])
            a[: j + 1] = s

        sbar += a

    sbar /= ntraces

    t_short = t[: lc]
    spec, fb = fftrl(sbar, t_short, 0, n)
    specb = 1.0 / np.abs(spec)

    # Parseval normalisation
    norm_in = np.linalg.norm(trin)
    norm_sp = np.linalg.norm(specb)
    if norm_sp != 0:
        specb = specb * norm_in / norm_sp

    return specb, fb
