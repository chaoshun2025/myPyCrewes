"""convm.py – causal convolution truncated to the length of the first argument.

Direct port of Margrave toolbox ``convm.m``.

The output is the same length as the first input *a*, with a half-Margrave
taper (``mwhalf``) applied at the trailing end to suppress the wrap-around
artefact from truncation.
"""

import numpy as np
from scipy.signal import lfilter
from mwhalf import mwhalf


def convm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Causal convolution of *a* with wavelet *b*, output length = len(a).

    Equivalent to MATLAB's ``filter(b, 1, a)`` (FIR filtering), so *b* is
    assumed to be **causal** (i.e. its first sample is at time zero).  For
    non-causal wavelets use ``convz``.

    The trailing end of the output is tapered with a 4 %-half-Margrave window
    to suppress the truncation artefact, exactly as in the original MATLAB.

    Parameters
    ----------
    a : array-like, 1-D **or** 2-D
        Signal(s) to be filtered.  If 2-D the convention follows the MATLAB
        original: if there are **more rows than columns** the matrix is
        treated as column-major (each column is one trace), otherwise as
        row-major (each row is one trace).  The output always has the same
        orientation as the input.
    b : array-like, 1-D
        Wavelet (FIR filter coefficients).  Must be no longer than *a*.

    Returns
    -------
    c : np.ndarray
        Filtered output, same shape as *a*.

    Raises
    ------
    ValueError
        If *b* is longer than *a*.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float).ravel()

    # ---- orientation logic mirrors MATLAB's [rows,cols]=size(a) ----
    # If rows > cols we work on the transpose (each row becomes a trace)
    transposed = False
    if a.ndim == 2 and a.shape[0] > a.shape[1]:
        a = a.T
        transposed = True

    # After possible transpose: a is (nvecs, na)  or  (na,) for 1-D
    single = a.ndim == 1
    if single:
        a = a[np.newaxis, :]   # (1, na)

    nvecs, na = a.shape
    nb = len(b)

    if na < nb:
        raise ValueError("First vector (a) must be longer than the wavelet (b).")

    # Zero-pad last sample (mirrors MATLAB: a(na)=0 before filter)
    # Make a writable copy so we can modify in place
    a = a.copy()
    if na > 1:
        a[:, -1] = 0.0

    mw = mwhalf(na, 4)   # trailing taper, matches MATLAB mwhalf(na,4)

    c = np.zeros_like(a)
    for k in range(nvecs):
        # scipy lfilter(b, [1], x) is identical to MATLAB filter(b, 1, x)
        c[k] = lfilter(b, [1.0], a[k]) * mw

    if single:
        c = c[0]          # back to 1-D
    elif transposed:
        c = c.T           # restore original orientation

    return c
