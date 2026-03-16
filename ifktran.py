"""ifktran.py – 2-D inverse f-k (frequency-wavenumber) transform.

Mirrors the MATLAB ``ifktran`` from the Margrave / CREWES seismic toolbox.
The forward transform is performed by :func:`fktran`.
"""

import numpy as np
from ifftrl import ifftrl
from mwindow import mwindow


def ifktran(spec, f, kx, nfpad=None, nkpad=None, percent=0.0):
    """
    2-D inverse f-k transform.

    Parameters
    ----------
    spec : np.ndarray (complex), shape (nf, nkx)
        Complex f-k spectrum as returned by :func:`fktran`.
    f : array_like, length nf
        Frequency coordinate vector (Hz).
    kx : array_like, length nkx
        Wavenumber coordinate vector (cycles/m).
    nfpad : int or None, optional
        Zero-pad the frequency axis to this length.
        Default: next power of 2 >= ``len(f)``.
    nkpad : int or None, optional
        Zero-pad the wavenumber axis to this length.
        Default: next power of 2 >= ``len(kx)``.
    percent : float, optional
        Raised-cosine taper length as a percentage of each axis length,
        applied before zero-padding.  Default 0 (no taper).

    Returns
    -------
    seis : np.ndarray (float), shape (nt, nx)
        Reconstructed 2-D seismic matrix (one trace per column).
    t : np.ndarray
        Time coordinate vector (s).
    x : np.ndarray
        Space coordinate vector (m).

    Notes
    -----
    * If *kx* appears to be unwrapped (i.e., ``kx[0] < 0``), the axis is
      re-wrapped to the natural FFT ordering before the inverse FFT, matching
      the MATLAB behaviour.
    * The kx→x transform uses ``np.fft.fft`` along rows (matching MATLAB's
      ``fft`` on the transposed spectrum).
    """
    spec = np.asarray(spec, dtype=complex)
    f = np.asarray(f, dtype=float).ravel()
    kx = np.asarray(kx, dtype=float).ravel()

    nf, nkx = spec.shape

    if len(f) != nf:
        raise ValueError("Frequency coordinate vector length incorrect")
    if len(kx) != nkx:
        raise ValueError("Wavenumber coordinate vector length incorrect")

    if nfpad is None or nfpad == 0:
        nfpad = int(2 ** np.ceil(np.log2(nf)))
    if nkpad is None or nkpad == 0:
        nkpad = int(2 ** np.ceil(np.log2(nkx)))

    # Re-wrap kx if it looks unwrapped (kx[0] < 0)
    if kx[0] < 0:
        ind = np.where(kx >= 0)[0]
        wrap_order = np.concatenate([ind, np.arange(ind[0])])
        kx = kx[wrap_order]
        spec = spec[:, wrap_order]

    # Optional taper in kx
    if percent > 0:
        # After possible re-wrap, apply taper in the original-length kx
        mw = mwindow(nkx, percent)
        # Re-apply the wrap ordering to the window
        if kx[0] < 0:  # already re-wrapped above; this branch won't be hit
            mw = mw[wrap_order]
        spec = spec * mw[np.newaxis, :]

    if nkx < nkpad:
        nkx_use = nkpad
    else:
        nkx_use = nkx

    # kx → x transform: MATLAB uses fft on rows
    specfx = np.fft.fft(spec, n=nkx_use, axis=1)

    # f → t transform using ifftrl
    seis, t = ifftrl(specfx, f)

    # Build x axis
    dkx = kx[1] - kx[0] if len(kx) > 1 else 1.0
    xmax = 1.0 / dkx
    dx = xmax / nkx_use
    x = np.arange(nkx_use) * dx

    return seis, t, x
