"""
deconw_stack.py  –  Apply Wiener deconvolution to a stacked section.

Direct port of the CREWES MATLAB function deconw_stack.m (Margrave).

Dependencies
------------
deconw.py, convm.py, near.py, mwindow.py
"""

import numpy as np

from deconw import deconw
from convm import convm
from near import near
from mwindow import mwindow


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def deconw_stack(
    stack: np.ndarray,
    t: np.ndarray,
    itr: int,
    tstart: float,
    tend: float,
    top: float = 0.1,
    stab: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Wiener (Levinson-Toeplitz) deconvolution to every trace in a
    stacked section.

    Parameters
    ----------
    stack : np.ndarray, shape (nsamp, ntr)
        Stacked section; each column is one trace, samples along rows.
    t : np.ndarray, 1-D
        Time coordinate vector, length = nsamp.
    itr : int
        * ``0``   – trace-by-trace mode: each trace designs its own operator.
        * ``1..ntr`` – single-operator mode: operator is designed from trace
          ``itr`` (1-based, matching MATLAB) and applied to all traces.
    tstart : float
        Start of the deconvolution design window (same units as *t*).
    tend : float
        End of the deconvolution design window (same units as *t*).
    top : float, optional
        Length of the deconvolution operator in **seconds**.  Converted to
        samples as ``round(top / dt)``.  Default = 0.1 s.
    stab : float, optional
        White-noise stabilisation factor.  Default = 1e-3.

    Returns
    -------
    stackd : np.ndarray, shape (nsamp, ntr)
        Deconvolved section (zero traces left unchanged).
    d : np.ndarray, 1-D
        Deconvolution operator of length ``nop``.  In trace-by-trace mode
        this is the average operator over all non-zero traces.  In
        single-operator mode it is the operator from the design trace.
        The estimated wavelet is::

            w = np.real(np.fft.ifft(1.0 / np.fft.fft(d)))

    Raises
    ------
    ValueError
        If *t* length does not match the first dimension of *stack*, or
        if *itr* is outside the valid range.
    """
    stack = np.asarray(stack, dtype=float)
    t     = np.asarray(t,     dtype=float).ravel()
    nsamp, ntr = stack.shape

    if len(t) != nsamp:
        raise ValueError("Length of t must equal the number of rows in stack.")

    # ------------------------------------------------------------------
    # Operator length in samples
    # MATLAB: dt = t(2)-t(1);  nop = round(top/dt)
    # ------------------------------------------------------------------
    dt  = t[1] - t[0]
    nop = int(np.round(top / dt))

    stackd = np.zeros_like(stack)
    small  = 1000.0 * np.finfo(float).eps
    ievery = 200                                    # progress report interval

    # Design-window indices and taper
    # MATLAB: idesign = near(t, tstart, tend)
    #         mw      = mwindow(length(idesign), 10)
    idesign = near(t, tstart, tend)
    mw      = mwindow(len(idesign), 10)

    # ------------------------------------------------------------------
    if itr == 0:
        # Trace-by-trace mode
        # MATLAB: d = zeros(nop,1)
        #         for k=1:ntr  [stackd(:,k),d2] = deconw(...)  d = d+d2  end
        #         d = d/n
        d_accum = np.zeros(nop)
        n_valid = 0

        for k in range(ntr):
            col = stack[:, k]
            if np.sum(np.abs(col)) > small:
                trdsign_win  = col[idesign] * mw
                col_d, d2    = deconw(col, trdsign_win, nop, stab)
                stackd[:, k] = col_d
                n_valid     += 1
                d_accum     += d2

            if (k + 1) % ievery == 0:
                print(f"finished trace {k + 1} out of {ntr}")

        if n_valid == 0:
            raise RuntimeError("All traces in the design window are zero.")
        d = d_accum / n_valid

    elif 1 <= itr <= ntr:
        # Single-operator mode (1-based trace index, matching MATLAB)
        # MATLAB: [tmp, d] = deconw(stack(:,itr), ...)
        #         for k=1:ntr  stackd(:,k) = convm(stack(:,k), d)  end
        k_design    = itr - 1                       # convert to 0-based
        col_design  = stack[:, k_design]
        trdsign_win = col_design[idesign] * mw
        _, d        = deconw(col_design, trdsign_win, nop, stab)

        for k in range(ntr):
            col = stack[:, k]
            if np.sum(np.abs(col)) > small:
                stackd[:, k] = convm(col, d)

            if (k + 1) % ievery == 0:
                print(f"finished trace {k + 1} out of {ntr}")

    else:
        raise ValueError(
            f"itr must be 0 (trace-by-trace) or between 1 and {ntr} (single operator)."
        )

    return stackd, d
