"""fktran.py – 2-D forward f-k (frequency-wavenumber) transform.

Mirrors the MATLAB ``fktran`` from the Margrave / CREWES seismic toolbox.
"""

import numpy as np
from fftrl import fftrl
from mwindow import mwindow


def fktran(seis, t, x, ntpad=None, nxpad=None, percent=0.0, ishift=1):
    """
    2-D forward f-k transform of a real-valued seismic matrix.

    Only positive frequencies are retained (one-sided spectrum); all
    wavenumbers are computed.  The inverse is :func:`ifktran`.

    Parameters
    ----------
    seis : np.ndarray, shape (nt, nx)
        Input 2-D seismic matrix.  One trace per column.
    t : array_like, length nt
        Time coordinate vector (s).
    x : array_like, length nx
        Space coordinate vector (m).
    ntpad : int or None, optional
        Zero-pad the time axis to this length before the FFT.
        Default: next power of 2 >= ``len(t)``.
    nxpad : int or None, optional
        Zero-pad the spatial axis to this length before the FFT.
        Default: next power of 2 >= ``len(x)``.
    percent : float, optional
        Raised-cosine taper length as a percentage of the axis length,
        applied before zero-padding.  Default 0 (no taper).
    ishift : int, optional
        If 1 (default), shift the k axis so that kx=0 is in the centre
        (``np.fft.fftshift``).  If 0, leave the natural FFT ordering.

    Returns
    -------
    spec : np.ndarray (complex), shape (nf, nx_padded)
        Complex f-k spectrum.
    f : np.ndarray
        Frequency coordinate vector (Hz).
    kx : np.ndarray
        Wavenumber coordinate vector (cycles/m).

    Notes
    -----
    The spatial FFT uses ``np.fft.ifft`` along rows, matching the MATLAB
    convention (``ifft`` on rows after the t→f transform).
    """
    seis = np.asarray(seis, dtype=float)
    t = np.asarray(t, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()

    nsamp, ntr = seis.shape

    if len(t) != nsamp:
        raise ValueError("Time coordinate vector length does not match rows of seis")
    if len(x) != ntr:
        raise ValueError("Space coordinate vector length does not match columns of seis")

    if ntpad is None or ntpad == 0:
        ntpad = int(2 ** np.ceil(np.log2(nsamp)))
    if nxpad is None or nxpad == 0:
        nxpad = int(2 ** np.ceil(np.log2(ntr)))

    # t → f transform (one-sided spectrum) with optional taper
    specfx, f = fftrl(seis, t, percent, ntpad)
    # specfx shape: (nf, ntr)

    # Spatial taper and pad
    if percent > 0:
        mw = mwindow(ntr, percent)
        specfx = specfx * mw[np.newaxis, :]

    if ntr < nxpad:
        ntr_use = nxpad
    else:
        ntr_use = ntr

    # kx–x transform: MATLAB uses ifft on rows (axis=1 in Python)
    # np.fft.ifft along axis=1 with ntr_use points
    spec = np.fft.ifft(specfx, n=ntr_use, axis=1)

    # Build kx axis
    dx = x[1] - x[0]
    kxnyq = 1.0 / (2.0 * dx)
    dkx = 2.0 * kxnyq / ntr_use
    kx_pos = np.arange(0, kxnyq, dkx)
    kx_neg = np.arange(-kxnyq, 0, dkx)
    kx = np.concatenate([kx_pos, kx_neg])

    if ishift:
        ikx = np.argsort(kx, kind="stable")
        kx = kx[ikx]
        spec = spec[:, ikx]

    return spec, f, kx
