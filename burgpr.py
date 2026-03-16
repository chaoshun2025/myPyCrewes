"""
burgpr.py  –  Burg prediction-error filter (unit lag).

Direct port of Margrave's ``burgpr.m`` (CREWES toolbox, May 1991),
adapted from Claerbout 1976: *Fundamentals of Geophysical Data Processing*,
p. 137.
"""

import numpy as np


def burgpr(trin: np.ndarray, lc: int) -> np.ndarray:
    """
    Compute an ``lc``-length Burg prediction-error filter (unit lag).

    Parameters
    ----------
    trin : array_like, 1-D
        Input trace used for filter design.
    lc : int
        Number of points in the prediction-error filter (operator length).
        Must be <= len(trin).

    Returns
    -------
    pfilt : np.ndarray, shape (lc,)
        Burg prediction-error filter.  The leading coefficient is 1.0
        (minimum-phase, causal).

    Raises
    ------
    ValueError
        If ``lc > len(trin)``.

    Notes
    -----
    The algorithm follows Claerbout (1976) exactly as coded by Margrave.
    Internally the filter coefficients are built up order-by-order using
    Burg's recursion:

        a_new[k] = a[k] - c[j] * conj(a[j-k])   for k = 1 … j

    where ``c[j]`` is the j-th Burg (Parcor / reflection) coefficient
    estimated from the forward/backward residuals ``ep``/``em``.
    """
    trin = np.asarray(trin, dtype=complex).ravel()
    lx = len(trin)

    if lc > lx:
        raise ValueError(
            f"Filter length lc={lc} exceeds trace length {lx}. "
            "Please check input parameters again."
        )

    # Initialise filter coefficient vector (MATLAB: a=[1.0 zeros(1,lc-1)])
    a = np.zeros(lc, dtype=complex)
    a[0] = 1.0

    # Reflection coefficients (MATLAB: c=zeros(1,lc))
    c = np.zeros(lc, dtype=complex)

    # Forward and backward residuals (MATLAB: em=trin; ep=trin)
    em = trin.copy()
    ep = trin.copy()

    # Running copy of 's' (last updated filter; returned as pfilt)
    s = np.zeros(lc, dtype=complex)

    # -----------------------------------------------------------------------
    # Main Burg recursion  (MATLAB loop: for j=2:lc)
    # -----------------------------------------------------------------------
    for j in range(1, lc):          # j runs 1 … lc-1  (0-based)
        # MATLAB:  ep(j:lx)    →  Python ep[j:]
        #          em(1:lx-j+1) →  Python em[:lx-j]
        ep_tail = ep[j:]            # length lx-j
        em_head = em[: lx - j]     # length lx-j

        # Denominator  (MATLAB: bot = ep(j:lx)*ep(j:lx)' + em(1:lx-j+1)*em(1:lx-j+1)')
        bot = (np.dot(ep_tail, ep_tail.conj())
               + np.dot(em_head, em_head.conj())).real

        # Numerator  (MATLAB: top = ep(j:lx)*em(1:lx-j+1)')
        top = np.dot(ep_tail, em_head.conj())

        # Reflection coefficient  (MATLAB: c(j) = 2*top/bot)
        if bot == 0.0:
            c[j] = 0.0
        else:
            c[j] = 2.0 * top / bot

        # Update residuals
        # MATLAB:
        #   epp = [ep(1:j-1)  ep(j:lx) - c(j)*em(1:lx-j+1)]
        #   em  = [em(1:lx-j+1) - conj(c(j))*ep(j:lx)  em(lx-j+2:lx)]
        ep_new = ep.copy()
        ep_new[j:] = ep_tail - c[j] * em_head

        em_new = em.copy()
        em_new[: lx - j] = em_head - np.conj(c[j]) * ep_tail

        ep = ep_new
        em = em_new

        # Update filter coefficients
        # MATLAB:  s(1:j) = a(1:j) - c(j)*conj(a(j:-1:1))
        #          a(1:j) = s(1:j)
        # Note: MATLAB indices are 1-based, so a(1:j+1) in 0-based
        idx = j + 1                               # number of coefficients to update
        s[:idx] = a[:idx] - c[j] * np.conj(a[idx - 1:: -1])
        a[:idx] = s[:idx]

    pfilt = s.copy()

    # Return real array when input was real-valued
    if np.all(pfilt.imag == 0):
        pfilt = pfilt.real

    return pfilt
