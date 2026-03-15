"""
deconf.py  –  Frequency-domain (spectral) deconvolution.

Ports the CREWES MATLAB function deconf.m to Python/NumPy.
Helper utilities (padpow2, pad, convz, triangle, gaussian) are
implemented inline so the file is self-contained.
"""

import numpy as np
from scipy.signal import hilbert as sp_hilbert


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_pow2(n: int) -> int:
    return 1 << int(np.ceil(np.log2(n)))


def _padpow2(x: np.ndarray) -> np.ndarray:
    """Zero-pad x to the next power of 2."""
    n = _next_pow2(len(x))
    out = np.zeros(n, dtype=x.dtype)
    out[:len(x)] = x
    return out


def _pad(x: np.ndarray, target_len: int) -> np.ndarray:
    """Zero-pad or truncate x to target_len."""
    if len(x) >= target_len:
        return x[:target_len].copy()
    out = np.zeros(target_len, dtype=x.dtype)
    out[:len(x)] = x
    return out


def _convz(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Zero-phase (full) convolution; returns array of length len(x)."""
    full = np.convolve(x, h, mode='full')
    # centre the result (same as MATLAB convz which is linear conv)
    return full[:len(x)]


def _triangle(n: int) -> np.ndarray:
    """Triangular window of length n (peak at centre)."""
    half = (n + 1) // 2
    w = np.concatenate([np.arange(1, half + 1), np.arange(half - 1, 0, -1)])
    return w[:n].astype(float)


def _gaussian_win(n: int) -> np.ndarray:
    """Gaussian window of length n."""
    sigma = n / 6.0
    t = np.arange(n) - n // 2
    return np.exp(-0.5 * (t / sigma) ** 2)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def deconf(
    trin: np.ndarray,
    trdsign: np.ndarray,
    n: int,
    stab: float = 1e-4,
    phase: int = 1,
    smoother: str = 'boxcar',
) -> tuple[np.ndarray, np.ndarray]:
    """
    Frequency-domain deconvolution of a seismic trace.

    Parameters
    ----------
    trin : np.ndarray
        Input trace to be deconvolved (1D).
    trdsign : np.ndarray
        Trace used for operator design (1D).
    n : int
        Number of points in the frequency-domain smoother.
    stab : float, optional
        Stabilisation factor as a fraction of the maximum power.
        Default = 1e-4.
    phase : int, optional
        0 → zero-phase whitening; 1 → minimum-phase deconvolution.
        Default = 1.
    smoother : str, optional
        Smoother type: 'boxcar' (default), 'triangle', or 'gaussian'.

    Returns
    -------
    trout : np.ndarray
        Deconvolved output trace (same length as trin).
    specinv : np.ndarray
        Inverse operator spectrum (complex, length = padded FFT length).
        The time-domain operator is ``np.real(np.fft.ifft(np.fft.ifftshift(specinv)))``.
    """
    trin = np.asarray(trin, dtype=float).ravel()
    trdsign = np.asarray(trdsign, dtype=float).ravel()
    N = len(trin)

    # Pad to power of 2
    trin_pad = _padpow2(trin)
    trdsign_pad = _pad(trdsign, len(trin_pad))
    Npad = len(trin_pad)

    # Scale smoother length to account for padding
    n_scaled = int(Npad * n / N)

    # Power spectrum of design trace
    spec = np.fft.fftshift(np.fft.fft(trdsign_pad))
    power = np.real(spec) ** 2 + np.imag(spec) ** 2

    # Stabilise
    max_p = np.max(power)
    delta_p = stab * max_p
    power = power + delta_p

    # Build smoother (odd length)
    nn = 2 * (n_scaled // 2) + 1
    if smoother == 'boxcar':
        smoo = np.ones(nn)
    elif smoother == 'triangle':
        smoo = _triangle(nn)
    elif smoother == 'gaussian':
        smoo = _gaussian_win(nn)
    else:
        raise ValueError(f"Unknown smoother type: '{smoother}'")

    # Smooth the power spectrum
    power = _convz(power, smoo) / np.sum(np.abs(smoo))

    # Build inverse operator spectrum
    if phase == 1:
        # Minimum-phase: spectral factorisation via Hilbert transform
        logspec = sp_hilbert(0.5 * np.log(power))
        specinv = np.exp(-np.conj(logspec))
    else:
        # Zero-phase whitening
        specinv = power ** (-0.5)

    # Deconvolve
    specin = np.fft.fftshift(np.fft.fft(trin_pad))
    specout = specin * specinv
    trout_pad = np.real(np.fft.ifft(np.fft.ifftshift(specout)))

    # Unpad to original length
    trout = trout_pad[:N]

    return trout, specinv
