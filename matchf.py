import numpy as np


def matchf(trin, trdsign, t, fsmoth, flag=0):
    """
    MATCHF: design a frequency-domain match filter.

    mfilt, tm = matchf(trin, trdsign, t, fsmoth, flag)

    The match filter will always have time zero in the middle and
    should be applied with ``convz``.

    Parameters
    ----------
    trin : array_like
        Input trace to be matched to trdsign.
    trdsign : array_like
        Desired output trace (target).
    t : array_like
        Time coordinate vector for trin (trin and trdsign must have the
        same length).
    fsmoth : float
        Length of the frequency smoother in Hz.
    flag : int, optional
        0 -> smooth the filter spectrum before inverse transform.
        1 -> smooth the spectra of trin and trdsign before division.
        2 -> both 0 and 1.
        Default = 0.

    Returns
    -------
    mfilt : np.ndarray
        Output match filter (time domain), length = len(trin).
    tm : np.ndarray
        Time coordinate for mfilt, centred on 0.

    Notes
    -----
    Requires ``fftrl``, ``ifftrl``, ``boxf``, and ``convz`` to be
    importable from the same package.
    """
    from fftrl import fftrl
    from ifftrl import ifftrl
    from boxf import boxf
    from convz import convz

    trin = np.asarray(trin, dtype=float).ravel()
    trdsign = np.asarray(trdsign, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()

    Trin, f = fftrl(trin, t)
    Trdsign, _ = fftrl(trdsign, t)

    if flag != 0:
        box = boxf(fsmoth, f)
        Trin = convz(Trin, box)
        Trdsign = convz(Trdsign, box)

    Mf = Trdsign / Trin

    if flag != 1:
        Mf = convz(Mf, boxf(fsmoth, f))

    mfilt = ifftrl(Mf, f)
    mfilt = np.fft.fftshift(mfilt)

    dt = t[1] - t[0]
    n = len(mfilt)
    tm = np.arange(n) * dt - (n // 2) * dt  # centred on 0

    return mfilt, tm
