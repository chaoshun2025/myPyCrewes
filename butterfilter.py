import numpy as np
from scipy.signal import butter, filtfilt, lfilter


def butterfilter(s, t, fmin=0.0, fmax=0.0, order=4, phase=1,
                 minlength=512, padwith='zeros'):
    """
    BUTTERFILTER: Butterworth bandpass, high-pass, and low-pass filtering.

    sf = butterfilter(s, t, ...)

    Parameters
    ----------
    s : array_like
        Seismic trace (1-D) or gather (2-D, one trace per column).
    t : array_like
        Time coordinate vector for s.
    fmin : float, optional
        Filter low cut (Hz).  Enter 0 for a low-pass filter.  Default = 0.
    fmax : float, optional
        Filter high cut (Hz).  Enter 0 for a high-pass filter.  Default = 0.
    order : int, optional
        Butterworth order.  Higher order = sharper roll-off.  Default = 4.
    phase : int, optional
        1 -> minimum phase (forward filter only).
        0 -> zero phase   (forward + backward via ``filtfilt``).
        Default = 1.
    minlength : int, optional
        Traces shorter than this are zero-padded; pad is removed on output.
        Default = 512.
    padwith : str, optional
        Padding strategy: ``'zeros'``, ``'mean'``, or ``'trend'``.
        Default = ``'zeros'``.

    Returns
    -------
    sf : np.ndarray
        Filtered trace(s), same shape as s.
    """
    s = np.asarray(s, dtype=float)
    t = np.asarray(t, dtype=float).ravel()
    nt = len(t)
    dt = t[1] - t[0]
    fn = 0.5 / dt  # Nyquist

    if not (0 <= fmin <= fn):
        raise ValueError("invalid fmin specification (outside 0 to Nyquist)")
    if not (0 <= fmax <= fn):
        raise ValueError("invalid fmax specification (outside 0 to Nyquist)")
    if padwith not in ('zeros', 'mean', 'trend'):
        raise ValueError("padwith must be 'zeros', 'mean', or 'trend'")
    if phase not in (0, 1):
        raise ValueError("phase must be 0 or 1")
    order = int(round(order))
    if order <= 0:
        raise ValueError("order must be a positive integer")

    # ---- normalise to 2-D column-major (nt x ntraces) ----
    transpose = False
    if s.ndim == 1:
        s = s[:, np.newaxis]
        transpose = True
    else:
        nr, nc = s.shape
        if nc == nt and nr == 1:
            s = s.T
            transpose = True
        elif nc == nt and nr > 1:
            raise ValueError("multiple traces should be stored in columns")
        elif nr != nt and nc != nt:
            raise ValueError("t and s have incompatible sizes")

    ntraces = s.shape[1]

    # ---- zero-pad if too short ----
    npad = 0
    if nt < minlength:
        npad = minlength - nt
        pad = np.zeros((npad, ntraces))
        if padwith == 'mean':
            for k in range(ntraces):
                pad[:, k] = np.mean(s[:, k])
        elif padwith == 'trend':
            t2 = np.arange(1, npad + 1) * dt + t[-1]
            for k in range(ntraces):
                p = np.polyfit(t, s[:, k], 1)
                pad[:, k] = np.polyval(p, t2)
        s = np.vstack([s, pad])

    # ---- design Butterworth filter ----
    W1, W2 = fmin / fn, fmax / fn
    if W1 == 0:
        b, a = butter(order, W2, btype='low')
    elif W2 == 0:
        b, a = butter(order, W1, btype='high')
    else:
        b, a = butter(order, [W1, W2], btype='band')

    # ---- apply ----
    if phase == 0:
        sf = filtfilt(b, a, s, axis=0)
    else:
        sf = lfilter(b, a, s, axis=0)

    if npad > 0:
        sf = sf[:nt, :]

    if transpose:
        sf = sf.ravel()

    return sf
