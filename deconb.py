"""
deconb.py  –  Burg-scheme (maximum entropy) deconvolution.

Direct port of Margrave's ``deconb.m`` (CREWES toolbox, May 1991).

Dependencies
------------
burgpr  – ``burgpr.py``  (Burg prediction-error filter)
convm   – ``convm.py``   (causal, same-length convolution)
balans  – ``balans.py``  (RMS energy balance)
"""

import numpy as np
from burgpr import burgpr
from convm import convm
from balans import balans


def deconb(
    trin: np.ndarray,
    trdsign: np.ndarray,
    l: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Burg-scheme deconvolution of an input trace.

    Parameters
    ----------
    trin : array_like, 1-D
        Input trace to be deconvolved.
    trdsign : array_like, 1-D
        Trace used for operator design (need not be the same as *trin*;
        e.g. a windowed portion of the trace or an ensemble average).
    l : int
        Prediction-error filter length (= length of the inverse operator).
        Must be <= len(trdsign).

    Returns
    -------
    trout : np.ndarray, 1-D
        Deconvolved output trace, same length as *trin*.
    pefilt : np.ndarray, 1-D, shape (l,)
        The prediction-error (inverse) operator used to deconvolve *trin*.

    Notes
    -----
    The algorithm mirrors ``deconb.m`` exactly:

    1. Estimate the PEF from *trdsign* via Burg's method (``burgpr``).
    2. Convolve the PEF with *trin* using causal same-length convolution
       (``convm``).
    3. Balance the output energy to match *trin* (``balans``).

    The MATLAB original handles both row- and column-vector inputs by
    transposing internally; this Python version accepts 1-D arrays
    (or any shape that ``.ravel()`` can flatten) and always returns 1-D
    output.
    """
    trin    = np.asarray(trin,    dtype=float).ravel()
    trdsign = np.asarray(trdsign, dtype=float).ravel()

    # Step 1 – design the prediction-error filter from the design trace
    # MATLAB: pefilt = burgpr(trdsign, l)'   (transposed to column, irrelevant in 1-D)
    pefilt = burgpr(trdsign, l)             # shape (l,)

    # Step 2 – convolve the PEF with the input trace (causal, same-length)
    # MATLAB: trout = convm(trin, pefilt)
    trout = convm(trin, pefilt)

    # Step 3 – balance output energy to match the input
    # MATLAB: trout = balans(trout, trin)
    trout = balans(trout, trin)

    return trout, pefilt
