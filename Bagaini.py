"""Bagaini.py – piecewise-constant reference-velocity model for PSPI migration.

Mirrors the MATLAB ``Bagaini`` from the Ferguson / Margrave CREWES toolbox.

Reference
---------
Bagaini et al. – reference-velocity selection for PSPI.  See also
Ferguson & Margrave (2005), Geophysics 70(5), S101–S109.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Internal helper functions (mirrors of the MATLAB sub-functions)
# ---------------------------------------------------------------------------

def _admit_vels(v, L):
    """
    Build a uniform grid of L+1 candidate velocities spanning [min(v), max(v)].

    Mirrors MATLAB ``admit_vels``.
    """
    cmin = np.min(v)
    cmax = np.max(v)
    dv   = (cmax - cmin) / L if L > 0 else 0.0
    c    = cmin + np.arange(L + 1) * dv
    return c


def _prob_den(c, v):
    """
    Probability density: fraction of velocity samples in each bin.

    Mirrors MATLAB ``prob_den``.

    Notes
    -----
    MATLAB original: ``P(k) = length(find(and(c(k)<=v, c(k+1)<=v)))``
    which counts samples where both ``c(k) <= v`` AND ``c(k+1) <= v``.
    That is equivalent to counting samples where ``v >= c(k+1)`` (the larger
    bound), i.e. samples in the *upper tail* of bin k.  This unusual
    definition is preserved here to stay faithful to the original.
    """
    L = len(c) - 1
    P = np.zeros(L)
    for k in range(L):
        # MATLAB: find(and(c(k)<=v, c(k+1)<=v))  →  v >= c[k+1]
        P[k] = np.sum(v >= c[k + 1])
    total = P.sum()
    if total > 0:
        P /= total
    return P


def _stat_ent(P):
    """
    Statistical entropy  S = -sum(P * log(P)),  ignoring zero entries.

    Mirrors MATLAB ``stat_ent``.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        logP = np.where(P > 0, np.log(P), 0.0)
    return -np.sum(P * logP)


def _binz(S, P):
    """
    Convert entropy ``S`` to a bin count, clamped to [1, len(P)].

    Mirrors MATLAB ``binz``.
    """
    return int(np.clip(round(np.exp(S) + 0.5), 1, len(P)))


def _bound_ref_vels(c, b, P):
    """
    Select ``b+1`` reference velocities from the candidate grid ``c`` using
    the cumulative probability ``P``.

    Mirrors MATLAB ``bound_ref_vels``.

    Notes
    -----
    The MATLAB code contains a typo: ``i/b`` instead of ``j/b`` in the inner
    loop.  Since ``i`` is undefined in that scope, MATLAB silently uses the
    imaginary unit (``i = sqrt(-1)``), making the condition always false.
    As a result, the inner loop body never executes and ``v`` remains all
    zeros except for ``v[0] = c[0]``.  This Python translation faithfully
    replicates that (buggy) behaviour: only ``v[0]`` is set from ``c[0]``
    and all other entries stay zero.  Callers that depend on the full
    reference-velocity set should be aware of this quirk.
    """
    nc  = len(c)
    # Cumulative probability breakpoints
    y       = np.zeros(nc)
    y[0]    = 0.0
    y[-1]   = 1.0
    for j in range(1, nc - 1):
        y[j] = np.sum(P[:j])

    v    = np.zeros(b + 1)
    v[0] = c[0]
    # The MATLAB inner condition `and(y(k)<j/b, i/b<=y(k+1))` uses the
    # imaginary unit `i` (a MATLAB bug), so the body never executes.
    # We replicate that: v[1:] remain 0.
    return v


def _piece_wise_itize(v, vel):
    """
    Map each sample in ``vel`` to the mean velocity of the bin it falls in.

    Mirrors MATLAB ``piece_wise_itize``.

    For each interval ``[v[j], v[j+1]]`` find all samples of ``vel`` in
    that range and replace them with the mean of those samples.  Samples
    outside all intervals keep the last value in ``v``.
    """
    v   = np.asarray(v,   dtype=float).ravel()
    vel = np.asarray(vel, dtype=float).ravel()

    out = np.full(len(vel), v[-1])   # default: last reference velocity

    for j in range(len(v) - 1):
        inds = np.where((vel >= v[j]) & (vel <= v[j + 1]))[0]
        if len(inds) > 0:
            out[inds] = np.mean(vel[inds])

    return out


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def Bagaini(bin_in, Lin, vel):
    """
    Build a piecewise-constant velocity model appropriate for PSPI migration.

    Parameters
    ----------
    bin_in : int
        Target maximum number of bins.  Typically ``size(vel, 2) - 1`` (i.e.
        ``nx - 1``).
    Lin : int
        Initial number of candidate velocity levels.  Try ``10``.
    vel : array_like, shape (nz, nx)
        Input velocity model (m/s).  Each row is a constant depth.

    Returns
    -------
    out : np.ndarray, shape (nz, nx)
        Piecewise-constant (blocked) velocity model.  Each row contains a
        laterally varying velocity that has been quantised to a small number
        of reference values.

    Notes
    -----
    The algorithm iterates over depths (rows) and for each row reduces the
    number of distinct velocity values to at most ``bin_in`` by an entropy-
    based binning procedure.  See Bagaini's original paper for details.

    The MATLAB implementation contains a bug in ``bound_ref_vels`` (use of
    the imaginary unit ``i`` instead of the loop variable ``j``), which
    causes all reference velocities beyond the first to be set to zero.
    This Python translation reproduces that behaviour faithfully.
    """
    vel = np.asarray(vel, dtype=float)
    nz, nx = vel.shape
    out = np.zeros((nz, nx))

    for j in range(nz):
        row = vel[j, :].copy()
        b   = nx
        L   = Lin
        c   = _admit_vels(row, L)

        # Iterate until the number of bins is within target
        for _ in range(nx, 0, -1):
            if b > bin_in:
                L -= 1
                if L < 1:
                    L = 1
                c = _admit_vels(row, L)
                P = _prob_den(c, row)
                S = _stat_ent(P)
                b = _binz(S, P)
            else:
                break

        P = _prob_den(c, row)
        v = _bound_ref_vels(c, b, P)
        out[j, :] = _piece_wise_itize(v, row)

    return out
