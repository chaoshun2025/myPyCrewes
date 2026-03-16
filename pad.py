"""
pad.py  –  Pad (or truncate) a 1-D trace to a target length.

Direct port of the CREWES MATLAB function pad.m (Margrave, June 1991).

Two modes
---------
flag=0  (default)
    Zeros appended to (or samples dropped from) the *end* of the trace.
flag=1
    The central sample of the input stays at the centre of the output;
    zeros fill symmetrically on both sides (or samples are removed
    symmetrically).  All four even/odd combinations of nin × nout are
    handled exactly as in the MATLAB source.

Usage notes
-----------
The second argument *trdsign* may be:

* a 1-D array  → target length = len(trdsign)
* an integer   → target length = trdsign  (convenient Python extension)
* a slice / range object whose length can be computed → same as int

This matches the way MATLAB callers pass either a vector or ``1:N``.
"""

import numpy as np


def _iseven(n: int) -> bool:
    return (n % 2) == 0


def pad(
    trin: np.ndarray,
    trdsign,
    flag: int = 0,
) -> np.ndarray:
    """
    Pad (or truncate) *trin* with zeros to match the length of *trdsign*.

    Parameters
    ----------
    trin : array-like, 1-D
        Input trace to be padded or truncated.
    trdsign : array-like **or** int
        Design vector whose length sets the target, **or** the target
        length given directly as an integer.
    flag : int, optional
        * ``0`` (default) – pad/truncate at the end.
        * ``1`` – keep the central sample centred; pad/truncate
          symmetrically.

    Returns
    -------
    trout : np.ndarray, 1-D
        Output trace of length ``nout``.

    Examples
    --------
    >>> pad([1, 2, 3, 4, 5], np.zeros(11), flag=1)
    array([0., 0., 0., 1., 2., 3., 4., 5., 0., 0., 0.])

    >>> pad([1, 2, 3, 4, 5], 3)
    array([1., 2., 3.])
    """
    trin = np.asarray(trin, dtype=float).ravel()
    nin  = len(trin)

    # Resolve nout from the second argument
    if isinstance(trdsign, (int, np.integer)):
        nout = int(trdsign)
    elif isinstance(trdsign, (range, slice)):
        # e.g. pad(x, range(N)) or pad(x, 1:N) style
        nout = len(trdsign) if isinstance(trdsign, range) else \
               len(range(*trdsign.indices(max(trdsign.start or 0,
                                              trdsign.stop or 0) + 1)))
    else:
        nout = len(np.asarray(trdsign).ravel())

    trout = np.zeros(nout, dtype=float)

    if flag == 0:
        # ----------------------------------------------------------------
        # Simple case: pad on end (or truncate)
        # MATLAB:  trout(1:nin)=trin   or   trout=trin(1:nout)
        # ----------------------------------------------------------------
        if nout >= nin:
            trout[:nin] = trin
        else:
            trout = trin[:nout].copy()

    else:
        # ----------------------------------------------------------------
        # Centred case: maintain position of central sample
        # Mirrors the MATLAB even/odd case table verbatim.
        # ----------------------------------------------------------------
        if _iseven(nin):
            if _iseven(nout):
                # even-in / even-out
                if nout > nin:
                    j0 = (nout - nin) // 2          # MATLAB: j0=(nout-nin)/2
                    trout[j0 : j0 + nin] = trin
                else:
                    k0 = (nin - nout) // 2          # MATLAB: k0=(nin-nout)/2
                    trout = trin[k0 : k0 + nout].copy()
            else:
                # even-in / odd-out
                if nout > nin:
                    j0 = (nout + 1) // 2 - nin // 2 - 1  # MATLAB: (nout+1)/2-nin/2-1
                    trout[j0 : j0 + nin] = trin
                else:
                    k0 = -(nout + 1) // 2 + nin // 2 + 1  # MATLAB: -(nout+1)/2+nin/2+1
                    trout = trin[k0 : k0 + nout].copy()
        else:
            if _iseven(nout):
                # odd-in / even-out
                if nout > nin:
                    j0 = nout // 2 + 1 - (nin + 1) // 2  # MATLAB: nout/2+1-(nin+1)/2
                    trout[j0 : j0 + nin] = trin
                else:
                    k0 = -nout // 2 - 1 + (nin + 1) // 2  # MATLAB: -nout/2-1+(nin+1)/2
                    trout = trin[k0 : k0 + nout].copy()
            else:
                # odd-in / odd-out
                if nout > nin:
                    j0 = (nout - nin) // 2          # MATLAB: (nout-nin)/2
                    trout[j0 : j0 + nin] = trin
                else:
                    k0 = (nin - nout) // 2          # MATLAB: (nin-nout)/2
                    trout = trin[k0 : k0 + nout].copy()

    return trout
