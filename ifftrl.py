"""ifftrl.py – inverse Fourier transform to a real trace.

Companion to :func:`fftrl`.  Reconstructs a real time-series from a
one-sided (positive-frequency) spectrum produced by :func:`fftrl`.
"""

import numpy as np


def ifftrl(spec, f):
    """
    Inverse Fourier transform returning a real time-series.

    Mirrors the MATLAB ``ifftrl`` from the Margrave toolbox: it builds the
    conjugate-symmetric two-sided spectrum and calls ``ifft``.

    Parameters
    ----------
    spec : array-like (complex), 1-D or 2-D
        One-sided spectrum as returned by :func:`fftrl`.
        For 2-D input the first axis is frequency (each column is one trace).
    f : array-like
        Frequency coordinate vector corresponding to *spec*.
        Only the last element (Nyquist) is used to derive dt.

    Returns
    -------
    r : np.ndarray (float)
        Real time-series.  Length = 2*(len(f)-1).
    t : np.ndarray
        Time coordinate vector.
    """
    spec = np.asarray(spec, dtype=complex)

    itr = False
    if spec.ndim == 1:
        nsamp = len(spec)
        ntr = 1
    elif spec.shape[0] == 1:       # row vector
        spec = spec.ravel()[:, np.newaxis]
        nsamp = spec.shape[0]
        ntr = 1
        itr = True
    else:
        nsamp, ntr = spec.shape[0], spec.shape[1] if spec.ndim > 1 else 1

    # Build conjugate-symmetric two-sided spectrum
    # MATLAB:  symspec = [spec(L1); conj(spec(L2))]
    #   L1 = 1:nsamp,  L2 = nsamp-1:-1:2  (1-based)
    L1 = np.arange(nsamp)
    L2 = np.arange(nsamp - 2, 0, -1)   # nsamp-2 down to 1 (0-based)

    if spec.ndim == 1:
        symspec = np.concatenate([spec[L1], np.conj(spec[L2])])
    else:
        symspec = np.vstack([spec[L1, :], np.conj(spec[L2, :])])

    r = np.real(np.fft.ifft(symspec, axis=0))

    # Time vector
    f = np.asarray(f, dtype=float)
    dt = 1.0 / (2.0 * f[-1])
    nt = r.shape[0]
    t = np.linspace(0.0, (nt - 1) * dt, nt)

    if itr:
        r = r.ravel()

    return r, t
