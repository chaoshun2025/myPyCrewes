"""
wwow.py  –  WWOW wavelet extraction (time-domain match filter).

"WWOW" stands for the method described in the CREWES toolbox.  It:
  1. Applies an amplitude envelope mask to isolate isolated events.
  2. Finds the constant phase rotation that maximises the spikiness
     of the masked analytic signal (Wiener criterion).
  3. Applies that rotation to get a pseudo-reflectivity.
  4. Designs a match filter (time domain) that matches the pseudo-
     reflectivity to the original trace.

Port of wwow.m from the CREWES MATLAB toolbox.

Depends on:
    hmask   (hmask.py)
    maxima  (maxima.py)
    phsrot  (phsrot.py)
    match   (match.py)
"""

import numpy as np
from hmask import hmask
from maxima import maxima
from phsrot import phsrot
from match import match


def wwow(
    trin: np.ndarray,
    t: np.ndarray,
    tleng: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    WWOW wavelet extraction via time-domain match filter.

    Parameters
    ----------
    trin : np.ndarray, 1D
        Input seismic trace.
    t : np.ndarray, 1D
        Time coordinate vector for trin.
    tleng : float
        Maximum length of output wavelet (seconds).

    Returns
    -------
    wavelet : np.ndarray, 1D
        Extracted wavelet (match filter).
    tw : np.ndarray, 1D
        Time coordinate vector for the wavelet (non-causal, centred at 0).
    pseudo : np.ndarray, 1D
        Pseudo-reflectivity used to design the match filter.
    """
    trin = np.asarray(trin, dtype=float).ravel()
    t    = np.asarray(t,    dtype=float).ravel()
    dt   = t[1] - t[0]

    # Step 1 – analytic signal and envelope mask
    mask, htrin = hmask(trin)

    # Step 2 – spline-refined maxima of the analytic signal.
    # maxima() refines peaks via cubic spline on real-valued traces,
    # so we pass real and imaginary parts separately and recombine.
    hsum_real = maxima(np.real(htrin), mask)
    hsum_imag = maxima(np.imag(htrin), mask)
    hsum = hsum_real + 1j * hsum_imag

    # Step 3 – constant phase rotation (Wiener criterion)
    top   = 2.0 * np.sum(np.real(hsum) * np.imag(hsum))
    bot   = np.sum(np.imag(hsum) ** 2 - np.real(hsum) ** 2)
    theta = 0.5 * np.arctan2(top, bot)          # radians

    # Test theta and theta + pi/2; keep the spikier rotation
    trot1 = phsrot(trin,  theta * 180.0 / np.pi).ravel()
    trot2 = phsrot(trin, (theta + np.pi / 2.0) * 180.0 / np.pi).ravel()
    rms1  = np.linalg.norm(trot1 * mask)
    rms2  = np.linalg.norm(trot2 * mask)
    if rms2 > rms1:
        theta = theta + np.pi / 2.0
        trot1 = trot2

    # Step 4 – pseudo-reflectivity
    pseudo = mask * trot1

    # Step 5 – time-domain match filter: conv(wavelet, pseudo) ≈ trin
    wavelet, tw = match(pseudo, trin, t, tleng, flag=0)

    # Build a centred time vector for the wavelet
    n  = len(wavelet)
    tw = np.arange(n) * dt - (n // 2) * dt

    return wavelet, tw, pseudo
