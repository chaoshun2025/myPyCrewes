"""
deconf.py  –  Frequency-domain (spectral) deconvolution.

Direct port of the CREWES MATLAB function deconf.m (Margrave, May 1991)
to Python/NumPy, using the companion helper modules padpow2.py, pad.py,
convz.py, and balans.py.

MATLAB source summary
----------------------
    N   = length(trin)          % original length before padding
    trin    = padpow2(trin)
    trdsign = pad(trdsign, trin)        % flag=0: match length of padded trin
    Npad = length(trin)
    n    = Npad * n / N                 % scale smoother for padded length

    spec   = fftshift(fft(trdsign))
    power  = real(spec)^2 + imag(spec)^2
    mean_p = sum(power) / length(power) % arithmetic mean
    power  = power + stab * mean_p

    smoother = boxcar(2*fix(n/2)+1)     % odd-length boxcar (ones)
    power    = convz(power, smoother)   % zero-phase (centred) convolution

    if phase == 1
        logspec = hilbert(0.5 * log(power))  % analytic signal of log-spectrum
        specinv = exp(-conj(logspec))         % min-phase inverse
    else
        specinv = power ^ (-0.5)              % zero-phase whitening

    specin  = fftshift(fft(trin))
    specout = specin .* specinv
    trout   = real(ifft(fftshift(specout)))
    trout   = balans(trout, trin)        % RMS-match output to input
    trout   = pad(trout, 1:N)           % truncate back to original length
"""

import numpy as np
from scipy.signal import hilbert as sp_hilbert

from padpow2 import padpow2
from pad import pad
from convz import convz
from balans import balans


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def deconf(
    trin: np.ndarray,
    trdsign: np.ndarray,
    n: int,
    stab: float = 1e-4,
    phase: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Frequency-domain (spectral) deconvolution of a seismic trace.

    Parameters
    ----------
    trin : array-like, 1-D
        Input trace to be deconvolved.
    trdsign : array-like, 1-D
        Trace used for operator design (may differ from *trin*, e.g.
        a windowed sub-trace).
    n : int
        Number of points in the frequency-domain **boxcar** smoother
        (applied to the power spectrum before inversion).  Automatically
        scaled for the padded FFT length.
    stab : float, optional
        Stabilisation factor expressed as a fraction of the **mean** power
        of the design spectrum.  Equivalent to adding white noise.
        Default = 1e-4.
    phase : int, optional
        * ``1`` (default) – minimum-phase deconvolution via spectral
          factorisation (Hilbert-transform method).
        * ``0`` – zero-phase whitening (divide by amplitude spectrum).

    Returns
    -------
    trout : np.ndarray, 1-D
        Deconvolved output trace, same length as *trin*.
    specinv : np.ndarray, 1-D (complex)
        Inverse-operator spectrum in fftshifted order, length = padded
        FFT size.  The time-domain operator is::

            d = np.real(np.fft.ifft(np.fft.ifftshift(specinv)))

        and the estimated wavelet is::

            w = np.real(np.fft.ifft(1.0 / np.fft.fft(d)))
    """
    trin    = np.asarray(trin,    dtype=float).ravel()
    trdsign = np.asarray(trdsign, dtype=float).ravel()

    # ------------------------------------------------------------------
    # 1. Pad traces to power of 2
    #    MATLAB: trin = padpow2(trin)
    #            trdsign = pad(trdsign, trin)   [flag=0 → end-pad]
    # ------------------------------------------------------------------
    N        = len(trin)
    trin_pad = padpow2(trin)                       # power-of-2 length
    Npad     = len(trin_pad)
    trdsign_pad = pad(trdsign, Npad)               # match padded length

    # ------------------------------------------------------------------
    # 2. Scale smoother length for padded FFT
    #    MATLAB: n = Npad * n / N
    # ------------------------------------------------------------------
    n_scaled = int(Npad * n / N)

    # ------------------------------------------------------------------
    # 3. Power spectrum of design trace
    #    MATLAB: spec  = fftshift(fft(trdsign))
    #            power = real(spec)^2 + imag(spec)^2
    # ------------------------------------------------------------------
    spec  = np.fft.fftshift(np.fft.fft(trdsign_pad))
    power = np.real(spec) ** 2 + np.imag(spec) ** 2

    # ------------------------------------------------------------------
    # 4. Stabilise: add stab * mean_power
    #    MATLAB: mean_p = sum(power)/length(power)  [arithmetic mean]
    #            power  = power + stab * mean_p
    # ------------------------------------------------------------------
    mean_p  = np.sum(power) / len(power)
    power   = power + stab * mean_p

    # ------------------------------------------------------------------
    # 5. Boxcar smoother (odd length) applied zero-phase
    #    MATLAB: smoother = boxcar(2*fix(n/2)+1)
    #            power    = convz(power, smoother)
    #    No normalisation — convz is a plain linear convolution,
    #    centred to keep the power spectrum aligned.
    # ------------------------------------------------------------------
    nn       = 2 * int(n_scaled // 2) + 1          # guaranteed odd
    smoother = np.ones(nn, dtype=float)             # boxcar
    # convz(r, w): nzero defaults to ceil((nw+1)/2), i.e. the centre of
    # the boxcar — exactly the zero-phase alignment MATLAB uses.
    # flag=0: suppress the mwindow taper (appropriate for traces, not spectra).
    power    = convz(power, smoother, flag=0)

    # ------------------------------------------------------------------
    # 6. Build inverse-operator spectrum
    # ------------------------------------------------------------------
    if phase == 1:
        # Minimum-phase via spectral factorisation
        # MATLAB: logspec = hilbert(0.5 * log(power))
        #         specinv = exp(-conj(logspec))
        logspec = sp_hilbert(0.5 * np.log(power))
        specinv = np.exp(-np.conj(logspec))
    else:
        # Zero-phase whitening
        # MATLAB: specinv = power ^ (-0.5)
        specinv = power ** (-0.5)

    # ------------------------------------------------------------------
    # 7. Deconvolve input trace
    #    MATLAB: specin  = fftshift(fft(trin))
    #            specout = specin .* specinv
    #            trout   = real(ifft(fftshift(specout)))
    # ------------------------------------------------------------------
    specin   = np.fft.fftshift(np.fft.fft(trin_pad))
    specout  = specin * specinv
    trout_pad = np.real(np.fft.ifft(np.fft.ifftshift(specout)))

    # ------------------------------------------------------------------
    # 8. RMS-balance output to input (MATLAB: balans)
    #    balans(trin, trref) scales trin to match RMS of trref.
    # ------------------------------------------------------------------
    trout_pad = balans(trout_pad, trin_pad)

    # ------------------------------------------------------------------
    # 9. Unpad back to original length
    #    MATLAB: trout = pad(trout, 1:N)   [flag=0 → simple truncation]
    # ------------------------------------------------------------------
    trout = pad(trout_pad, N)

    return trout, specinv
