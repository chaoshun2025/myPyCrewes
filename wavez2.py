"""
wavez2.py  –  Zero-phase bandpass FIR wavelet.

Uses scipy.signal.firwin (equivalent to MATLAB's fir1) to design a
linear-phase bandpass filter and normalises it so that its convolution
with a reference sinusoid at the dominant frequency has unit peak amplitude.

Port of wavez2.m from the CREWES MATLAB toolbox.

Note: the MATLAB source contains a bug – it references the variable
``fdom`` which is never defined as a parameter.  The Python version
derives the dominant frequency as the geometric mean of f1 and f2,
which is the natural choice for a symmetric bandpass.
"""

import numpy as np
from scipy.signal import firwin


def wavez2(
    f1: float,
    f2: float,
    tlength: float,
    t: np.ndarray,
    fdom: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Zero-phase bandpass FIR wavelet.

    Parameters
    ----------
    f1 : float
        Low-end cutoff frequency of the bandpass (Hz).
    f2 : float
        High-end cutoff frequency of the bandpass (Hz).
    tlength : float
        Temporal length of the wavelet (seconds).
    t : np.ndarray, 1D (at least 2 elements)
        Time coordinate vector that provides the sample interval
        (only ``t[1] - t[0]`` is used).
    fdom : float or None, optional
        Dominant frequency used for normalisation (Hz).
        Defaults to the geometric mean of f1 and f2.

    Returns
    -------
    wavelet : np.ndarray, 1D
        Zero-phase FIR wavelet of length n = round(tlength / dt) + 1.
    twave : np.ndarray, 1D
        Time coordinate vector centred at 0, length n.
    """
    t = np.asarray(t, dtype=float).ravel()
    dt   = t[1] - t[0]
    fnyq = 0.5 / dt

    if f1 <= 0 or f2 >= fnyq:
        raise ValueError(
            f"f1 must be > 0 and f2 must be < fnyq={fnyq:.2f} Hz."
        )
    if f2 <= f1:
        raise ValueError("f2 must be greater than f1.")

    n = int(round(tlength / dt)) + 1
    # firwin requires an odd filter length for a Type-I linear-phase filter
    if n % 2 == 0:
        n += 1

    # Design bandpass FIR filter (scipy equivalent of MATLAB fir1)
    wavelet = firwin(
        n,
        [f1 / fnyq, f2 / fnyq],
        pass_zero=False,   # bandpass
        window='hamming',
    )

    twave = np.linspace(-tlength / 2.0, tlength / 2.0, n)

    # Normalise: match peak amplitude of conv(refwave, wavelet) to refwave
    if fdom is None:
        fdom = np.sqrt(f1 * f2)     # geometric mean – natural dominant frequency

    refwave = np.sin(2.0 * np.pi * fdom * twave)
    # Same-length convolution (causal)
    reftest_full = np.convolve(refwave, wavelet)
    reftest = reftest_full[: len(refwave)]

    peak_ref  = np.max(np.abs(refwave))
    peak_test = np.max(np.abs(reftest))
    if peak_test > 0:
        wavelet = wavelet * (peak_ref / peak_test)

    return wavelet, twave
