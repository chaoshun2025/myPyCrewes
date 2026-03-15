"""todb.py – convert a complex spectrum to (dB, phase) representation."""

import numpy as np


def todb(S, refamp=None):
    """
    Convert a complex spectrum from (real, imag) to (dB, phase).

    The returned complex array encodes amplitude in dB as the real part
    and phase angle (radians) as the imaginary part, matching the MATLAB
    convention used by the Margrave seismic toolbox.

    Parameters
    ----------
    S : array-like (complex)
        Input complex spectrum.
    refamp : float, optional
        Reference amplitude for the dB computation.  Defaults to
        max(|S|) so all values are <= 0 dB.

    Returns
    -------
    Sdb : np.ndarray (complex)
        real(Sdb) = amplitude in dB relative to refamp.
        imag(Sdb) = phase angle in radians.
    """
    S = np.asarray(S, dtype=complex)
    amp = np.abs(S)
    phs = np.angle(S)

    if refamp is None:
        refamp = np.max(amp)

    amp_db = 20.0 * np.log10(amp / refamp)
    return amp_db + 1j * phs
