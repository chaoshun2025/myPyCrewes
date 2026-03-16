"""
deconb_stack.py  –  Apply Burg deconvolution to a stacked seismic section.

Direct port of Margrave's ``deconb_stack.m`` (CREWES toolbox, May 2016).

Dependencies
------------
deconb   – ``deconb.py``    (single-trace Burg deconvolution)
convm    – ``convm.py``     (causal same-length convolution)
near     – ``near.py``      (index search helper)
mwindow  – ``mwindow.py``   (Margrave taper window)
"""

import numpy as np
from deconb import deconb
from convm import convm
from near import near
from mwindow import mwindow


def deconb_stack(
    stack: np.ndarray,
    t: np.ndarray,
    itr: int,
    tstart: float,
    tend: float,
    top: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Burg deconvolution to all traces in a stacked seismic section.

    Parameters
    ----------
    stack : np.ndarray, shape (nt, ntr)
        Stacked section as a 2-D array of traces (samples × traces).
    t : np.ndarray, 1-D
        Time coordinate vector, length ``nt``.
    itr : int
        Trace number (1-based) to use for operator design, applied to the
        entire section.  Set to **0** for trace-by-trace (independent)
        deconvolution.
    tstart : float
        Start of the deconvolution design window (same units as *t*).
    tend : float
        End of the deconvolution design window (same units as *t*).
    top : float, optional
        Length of the deconvolution operator in seconds.
        Default is 0.1 s.

    Returns
    -------
    stackd : np.ndarray, shape (nt, ntr)
        Deconvolved section, same shape as *stack*.
    d : np.ndarray, 1-D
        Deconvolution operator.  In trace-by-trace mode (``itr == 0``)
        this is the average of all per-trace operators.  In single-operator
        mode it is the operator designed from trace ``itr``.

    Notes
    -----
    * A 10 %-taper Margrave window (``mwindow``) is applied to the design
      gate before passing it to ``deconb``, exactly as in the MATLAB original.
    * Zero traces (all-zero amplitude) are skipped to avoid division by zero;
      their output columns remain zero.
    * Progress is printed every 200 traces to match the MATLAB behaviour.
    * ``itr`` is **1-based** to mirror the MATLAB convention.  Valid values
      are 0 (trace-by-trace) or 1 … ntr.

    The estimated wavelet can be recovered as::

        w = np.fft.ifft(1.0 / np.fft.fft(d, n=len(t)))
    """
    stack = np.asarray(stack, dtype=float)
    t     = np.asarray(t,     dtype=float).ravel()

    nt, ntr = stack.shape

    if len(t) != nt:
        raise ValueError(
            f"Length of t ({len(t)}) does not match number of rows in stack ({nt})."
        )

    if itr < 0 or itr > ntr:
        raise ValueError(
            f"itr={itr} is out of range; must be 0 (trace-by-trace) or 1…{ntr}."
        )

    # Sample interval and operator length in samples
    dt  = t[1] - t[0]
    nop = int(round(top / dt))

    # Design-window sample indices (0-based, via near)
    idesign = near(t, tstart, tend)        # 0-based index array
    mw      = mwindow(len(idesign), 10)    # 10 % Margrave taper

    stackd = np.zeros_like(stack)
    small  = 1000.0 * np.finfo(float).eps
    ievery = 200

    # ------------------------------------------------------------------
    # Mode 0: trace-by-trace deconvolution
    # ------------------------------------------------------------------
    if itr == 0:
        d = np.zeros(nop)
        n = 0

        for k in range(ntr):
            col = stack[:, k]
            if np.sum(np.abs(col)) > small:
                trout, d2 = deconb(col, col[idesign] * mw, nop)
                stackd[:, k] = trout
                d += d2
                n += 1

            if (k + 1) % ievery == 0:
                print(f"finished trace {k + 1} out of {ntr}")

        if n > 0:
            d /= n

    # ------------------------------------------------------------------
    # Mode itr >= 1: design operator from one trace, apply to all
    # ------------------------------------------------------------------
    else:
        # Convert to 0-based index
        k_design = itr - 1
        design_col = stack[:, k_design]
        _, d = deconb(design_col, design_col[idesign] * mw, nop)

        for k in range(ntr):
            col = stack[:, k]
            if np.sum(np.abs(col)) > small:
                stackd[:, k] = convm(col, d)

            if (k + 1) % ievery == 0:
                print(f"finished trace {k + 1} out of {ntr}")

    return stackd, d
