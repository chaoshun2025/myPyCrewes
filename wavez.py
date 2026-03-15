"""
wavez.py  –  Zero-phase wavelet with a realistic (Ricker-like) amplitude
spectrum.

The MATLAB original calls CREWES helper functions:
  tntamp(fdom, f)  – "tent" amplitude spectrum (triangular in log-f)
  ifftrl(aspec, f) – inverse real-valued FFT from one-sided spectrum
  convz(a, b)      – zero-phase (full) convolution
  between(a, b, v) – find indices of v where a <= v <= b

All are implemented inline here so the file is self-contained.

Port of wavez.m from the CREWES MATLAB toolbox.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tntamp(fdom: float, f: np.ndarray) -> np.ndarray:
    """
    "Tent" (triangular) amplitude spectrum centred on *fdom*.

    Matches CREWES tntamp: linear ramp up from 0 to fdom, then linear ramp
    down from fdom to 2*fdom (and zero beyond).
    This produces a minimum-phase wavelet in the time domain whose dominant
    frequency is approximately fdom.
    """
    f    = np.asarray(f, dtype=float)
    aspec = np.zeros_like(f)
    # Rising edge: 0 → fdom
    mask1 = (f >= 0) & (f <= fdom)
    aspec[mask1] = f[mask1] / fdom
    # Falling edge: fdom → 2*fdom
    mask2 = (f > fdom) & (f <= 2.0 * fdom)
    aspec[mask2] = 2.0 - f[mask2] / fdom
    return aspec


def _ifftrl(aspec: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Inverse real-valued FFT from a one-sided (positive-frequency) spectrum.

    Reconstructs the real time-domain signal from the amplitude spectrum
    aspec defined on f = [0, fnyq].  The phase is set to zero (zero-phase).
    """
    nf   = len(aspec)
    # Number of time samples = 2*(nf-1)
    nfull = 2 * (nf - 1)
    # Build the full (two-sided) complex spectrum
    spec_full = np.zeros(nfull, dtype=complex)
    spec_full[0]       = aspec[0]
    spec_full[1: nf]   = aspec[1:] / 2.0    # positive freqs (halved for symmetry)
    spec_full[nf:]     = aspec[nf - 2: 0: -1] / 2.0  # negative freqs (mirror)
    wavelet = np.real(np.fft.ifft(spec_full)) * nfull
    return wavelet


def _convz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Zero-phase (same-length) convolution.  Returns array of len(a)."""
    full = np.convolve(a, b, mode='full')
    return full[: len(a)]


def _between(vmin: float, vmax: float, vec: np.ndarray) -> np.ndarray:
    """Return indices where vmin <= vec <= vmax."""
    return np.where((vec >= vmin) & (vec <= vmax))[0]


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def wavez(
    dt: float,
    fdom: float = 15.0,
    tlength: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Zero-phase wavelet with a realistic amplitude spectrum.

    Parameters
    ----------
    dt : float
        Desired temporal sample interval (seconds).
    fdom : float, optional
        Dominant frequency in Hz.  Default = 15 Hz.
    tlength : float or None, optional
        Wavelet length in seconds.  Default = 127 * dt (next power-of-2
        minus 1 samples, matching the MATLAB default).

    Returns
    -------
    wavelet : np.ndarray, 1D
        Zero-phase wavelet, trimmed to *tlength*.
    tw : np.ndarray, 1D
        Time coordinate vector for the wavelet (centred at 0).
    """
    if tlength is None:
        tlength = 127.0 * dt

    # Build a power-of-2 length time vector centred at 0
    nt    = int(tlength / dt) + 1
    nt    = 2 ** int(np.ceil(np.log2(nt)))
    tmax  = dt * nt / 2.0
    tw_full = np.arange(nt) * dt - tmax + dt   # centred, length nt
    # exact xcoord equivalent: start at -tmax, step dt, nt points
    tw_full = np.linspace(-tmax, -tmax + (nt - 1) * dt, nt)

    fnyq = 1.0 / (2.0 * dt)
    f    = np.linspace(0.0, fnyq, nt // 2 + 1)

    # Amplitude spectrum → inverse FFT
    aspec   = _tntamp(fdom, f)
    wav_raw = _ifftrl(aspec, f)               # length nt

    # Apply fftshift to centre the wavelet
    wavelet_full = np.fft.fftshift(wav_raw)

    # Normalise: scale so that the peak of conv(refwave, wavelet) == peak of refwave
    refwave = np.sin(2.0 * np.pi * fdom * tw_full)
    reftest = _convz(refwave, wavelet_full)
    peak_ref  = np.max(np.abs(refwave))
    peak_test = np.max(np.abs(reftest))
    if peak_test > 0:
        wavelet_full *= peak_ref / peak_test

    # Trim to requested length
    ind     = _between(-tlength / 2.0, tlength / 2.0, tw_full)
    wavelet = wavelet_full[ind]
    tw      = tw_full[ind]

    return wavelet, tw
