"""
wwowf.py  –  WWOW wavelet extraction (frequency-domain match filter).

Identical to wwow.py for the phase-estimation steps, but uses a
frequency-domain match filter (spectral division with smoothing) instead
of a time-domain match filter.

Port of wwowf.m from the CREWES MATLAB toolbox.

Depends on:
    hmask   (hmask.py)
    maxima  (maxima.py)
    phsrot  (phsrot.py)

The frequency-domain match filter (matchf) is implemented inline because
no separate matchf.py has been provided.
"""

import numpy as np
from hmask import hmask
from maxima import maxima
from phsrot import phsrot


# ---------------------------------------------------------------------------
# Inline frequency-domain match filter (matchf)
# ---------------------------------------------------------------------------

def _matchf(
    trin: np.ndarray,
    trdsign: np.ndarray,
    t: np.ndarray,
    fsmooth: float,
    flag: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Frequency-domain match filter.

    Estimates the wavelet W(f) such that  W(f) * TRIN(f) ≈ TRDSIGN(f)
    by stabilised spectral division with optional smoothing.

    Parameters
    ----------
    trin    : 1D array – input trace (pseudo-reflectivity)
    trdsign : 1D array – target trace
    t       : 1D array – time coordinate vector
    fsmooth : float    – frequency smoother width in Hz
    flag    : int
        0 → smooth the ratio (wavelet spectrum) before inverse FFT
        1 → smooth TRIN and TRDSIGN spectra before division
        2 → both 0 and 1

    Returns
    -------
    wavelet : 1D array – extracted wavelet (centred at t=0)
    tw      : 1D array – time coordinate vector
    """
    trin    = np.asarray(trin,    dtype=float).ravel()
    trdsign = np.asarray(trdsign, dtype=float).ravel()
    t       = np.asarray(t,       dtype=float).ravel()
    dt      = t[1] - t[0]
    N       = len(trin)

    TRIN    = np.fft.fft(trin)
    TRDSIGN = np.fft.fft(trdsign)

    df       = 1.0 / (N * dt)
    n_smooth = max(1, int(round(fsmooth / df)))

    def _boxcar(x: np.ndarray, n: int) -> np.ndarray:
        kernel = np.ones(n) / n
        return np.convolve(np.abs(x), kernel, mode='same')

    if flag == 1 or flag == 2:
        amp_trin = _boxcar(TRIN, n_smooth)
        stab     = 1e-4 * amp_trin.max()
        W        = (TRDSIGN * np.conj(TRIN)) / (amp_trin ** 2 + stab)
    else:
        amp2 = np.abs(TRIN) ** 2
        stab = 1e-4 * amp2.max()
        W    = (TRDSIGN * np.conj(TRIN)) / (amp2 + stab)

    if flag == 0 or flag == 2:
        W_abs_smooth = _boxcar(W, n_smooth)
        W = W_abs_smooth * np.exp(1j * np.angle(W))

    wavelet = np.real(np.fft.ifft(W))
    wavelet = np.fft.fftshift(wavelet)
    tw      = np.arange(N) * dt - (N // 2) * dt

    return wavelet, tw


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def wwowf(
    trin: np.ndarray,
    t: np.ndarray,
    fsmooth: float,
    flag: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    WWOW wavelet extraction via frequency-domain match filter.

    Parameters
    ----------
    trin : np.ndarray, 1D
        Input seismic trace.
    t : np.ndarray, 1D
        Time coordinate vector for trin.
    fsmooth : float
        Width of the frequency smoother in Hz.
    flag : int, optional
        0 → smooth the filter spectrum (default)
        1 → smooth the input spectra before division
        2 → both

    Returns
    -------
    wavelet : np.ndarray, 1D
        Extracted wavelet (centred at t=0).
    tw : np.ndarray, 1D
        Time coordinate vector for the wavelet.
    pseudo : np.ndarray, 1D
        Pseudo-reflectivity used to design the match filter.
    """
    trin = np.asarray(trin, dtype=float).ravel()
    t    = np.asarray(t,    dtype=float).ravel()

    # Step 1 – analytic signal and envelope mask
    mask, htrin = hmask(trin)

    # Step 2 – spline-refined maxima of the analytic signal
    hsum_real = maxima(np.real(htrin), mask)
    hsum_imag = maxima(np.imag(htrin), mask)
    hsum = hsum_real + 1j * hsum_imag

    # Step 3 – constant phase rotation (Wiener criterion)
    top   = 2.0 * np.sum(np.real(hsum) * np.imag(hsum))
    bot   = np.sum(np.imag(hsum) ** 2 - np.real(hsum) ** 2)
    theta = 0.5 * np.arctan2(top, bot)

    trot1 = phsrot(trin,  theta * 180.0 / np.pi).ravel()
    trot2 = phsrot(trin, (theta + np.pi / 2.0) * 180.0 / np.pi).ravel()
    rms1  = np.linalg.norm(trot1 * mask)
    rms2  = np.linalg.norm(trot2 * mask)
    if rms2 > rms1:
        theta = theta + np.pi / 2.0
        trot1 = trot2

    # Step 4 – pseudo-reflectivity
    pseudo = mask * trot1

    # Step 5 – frequency-domain match filter
    wavelet, tw = _matchf(pseudo, trin, t, fsmooth, flag)

    return wavelet, tw, pseudo
