"""
deconf_stack.py  –  Apply frequency-domain deconvolution to a stacked section.

Direct port of the CREWES MATLAB function deconf_stack.m (Margrave).

Dependencies
------------
deconf.py, convm.py, near.py, mwindow.py
"""

import numpy as np

from deconf import deconf
from convm import convm
from near import near
from mwindow import mwindow


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def deconf_stack(
    stack: np.ndarray,
    t: np.ndarray,
    itr: int,
    tstart: float,
    tend: float,
    fsmo: float = 0.1,
    stab: float = 1e-3,
    phase: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply frequency-domain deconvolution to every trace in a stacked section.

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
    fsmo : float, optional
        Length of the frequency smoother in Hz.  Default = 0.1.
    stab : float, optional
        White-noise stabilisation factor.  Default = 1e-3.
    phase : int, optional
        ``1`` = minimum-phase (default), ``0`` = zero-phase.

    Returns
    -------
    stackd : np.ndarray, shape (nsamp, ntr)
        Deconvolved section (zero traces left unchanged).
    specinv : np.ndarray, 1-D (complex)
        Inverse-operator spectrum.  In trace-by-trace mode this is the
        average over all non-zero traces.  In single-operator mode it is
        the spectrum from the design trace.
        The time-domain operator is::

            d = np.real(np.fft.ifft(np.fft.ifftshift(specinv)))

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
    # Smoother size in samples
    # MATLAB: df = 1/(t(end)-t(1));  nop = round(fsmo/df)
    # ------------------------------------------------------------------
    df  = 1.0 / (t[-1] - t[0])
    nop = int(np.round(fsmo / df))

    stackd = np.zeros_like(stack)
    small  = 1000.0 * np.finfo(float).eps

    # Design-window indices and taper
    # MATLAB: idesign = near(t, tstart, tend)
    #         mw      = mwindow(length(idesign), 10)
    idesign = near(t, tstart, tend)
    mw      = mwindow(len(idesign), 10)

    # ------------------------------------------------------------------
    if itr == 0:
        # Trace-by-trace mode
        # MATLAB: for k=1:ntr  [stackd(:,k), si] = deconf(...)  end
        #         specinv = average over all non-zero traces
        specinv = None
        n_valid = 0

        for k in range(ntr):
            col = stack[:, k]
            if np.sum(np.abs(col[idesign])) > small:
                trdsign_win = col[idesign] * mw
                col_d, si   = deconf(col, trdsign_win, nop, stab, phase)
                stackd[:, k] = col_d
                n_valid += 1
                specinv = si if n_valid == 1 else specinv + si

        if n_valid == 0:
            raise RuntimeError("All traces in the design window are zero.")
        specinv = specinv / n_valid

    elif 1 <= itr <= ntr:
        # Single-operator mode (1-based trace index, matching MATLAB)
        # MATLAB: [tmp, specinv] = deconf(stack(:,itr), ...)
        #         d = real(ifft(fftshift(specinv)))
        #         for k = 1:ntr  stackd(:,k) = convm(stack(:,k), d)  end
        k_design    = itr - 1                       # convert to 0-based
        col_design  = stack[:, k_design]
        trdsign_win = col_design[idesign] * mw
        _, specinv  = deconf(col_design, trdsign_win, nop, stab, phase)

        # Time-domain operator from the inverse spectrum
        # MATLAB: d = real(ifft(fftshift(specinv)))
        d = np.real(np.fft.ifft(np.fft.ifftshift(specinv)))

        for k in range(ntr):
            col = stack[:, k]
            if np.sum(np.abs(col)) > small:
                stackd[:, k] = convm(col, d)

    else:
        raise ValueError(
            f"itr must be 0 (trace-by-trace) or between 1 and {ntr} (single operator)."
        )

    return stackd, specinv
