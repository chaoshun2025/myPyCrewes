"""wavemin.py – minimum-phase wavelet for impulsive seismic sources.

Direct port of Margrave toolbox ``wavemin.m``.

Algorithm
---------
1. Build a power spectrum using :func:`tntamp`.
2. Convert to an autocorrelation via :func:`ifftrl`.
3. Solve the Toeplitz system T·w_inv = δ with :func:`levrec` to get the
   inverse filter.
4. The wavelet is the time-domain inverse of ``w_inv``.
5. Normalise so that a reference sine wave at *fdom* passes with unit
   amplitude after :func:`convm`.
"""

import numpy as np
from tntamp  import tntamp
from ifftrl  import ifftrl
from levrec  import levrec
from xcoord  import xcoord
from convm   import convm


def wavemin(dt: float, fdom: float = 15.0, tlength: float = None) -> tuple:
    """
    Create a minimum-phase wavelet for impulsive seismic sources.

    Parameters
    ----------
    dt : float
        Desired temporal sample rate (s).
    fdom : float, optional
        Dominant frequency (Hz).  Default 15 Hz.
    tlength : float, optional
        Wavelet length (s).  Default ``127 * dt`` (128 samples, power of 2).

    Returns
    -------
    wavelet : np.ndarray, 1-D
        Minimum-phase wavelet, normalised so a sine wave at *fdom* passes
        with unit amplitude through :func:`convm`.
    twave : np.ndarray, 1-D
        Time coordinate vector starting at 0.

    Notes
    -----
    The algorithm follows ``wavemin.m`` step by step:

    * ``nt = nextpow2(2*tlength/dt + 1)``  — internal FFT length (power of 2)
    * Power spectrum → autocorrelation via ``ifftrl``
    * Levinson recursion (``levrec``) on the first ``nlags = tlength/dt + 1``
      lags to obtain the inverse filter
    * Wavelet = ``real(ifft(1 / fft(w_inv)))``
    * Normalised by convolution with a reference sinusoid via ``convm``
    """
    if tlength is None:
        tlength = 127.0 * dt

    # ------------------------------------------------------------------ grid
    # MATLAB: nt = 2^nextpow2(2*tlength/dt + 1)
    nt_raw = 2.0 * tlength / dt + 1.0
    nt     = int(2 ** np.ceil(np.log2(nt_raw)))

    tw = np.arange(nt) * dt       # length nt

    # ------------------------------------------------------------------ freq
    fnyq = 1.0 / (2.0 * dt)
    # MATLAB: f = linspace(0, fnyq, length(tw)/2 + 1)
    nf = nt // 2 + 1
    f  = np.linspace(0.0, fnyq, nf)

    # ---------------------------------------------------------- power spectrum
    powspec = tntamp(fdom, f) ** 2

    # -------------------------------------------------------- autocorrelation
    # ifftrl(powspec, f) → real time-series of length nt
    auto, _ = ifftrl(powspec, f)

    # -------------------------------------------------- Levinson recursion
    # MATLAB: nlags = tlength/dt + 1;  b = [1 zeros(1,nlags-1)]'
    nlags = int(round(tlength / dt)) + 1
    b     = np.zeros(nlags)
    b[0]  = 1.0                      # unit spike as RHS

    winv = levrec(auto[:nlags], b)   # shape (nlags,)

    # ---------------------------------------------------------- invert wavelet
    # MATLAB: wavelet = real(ifft(1 ./ fft(winv)))
    # fft/ifft length = nlags (MATLAB does not zero-pad here)
    W_inv   = np.fft.fft(winv)
    wavelet = np.real(np.fft.ifft(1.0 / W_inv))

    # ---------------------------------------------------------- time coord
    # MATLAB: twave = xcoord(0, dt, wavelet)'
    twave = xcoord(0.0, dt, wavelet)

    # ---------------------------------------------------------- normalise
    # Scale so that convm(sin(2*pi*fdom*twave), wavelet) has unit amplitude
    refwave = np.sin(2.0 * np.pi * fdom * twave)
    reftest = convm(refwave, wavelet)
    fact    = np.max(refwave) / np.max(reftest)
    wavelet = wavelet * fact

    return wavelet, twave
