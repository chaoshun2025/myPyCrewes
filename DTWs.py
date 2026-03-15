"""
DTWs.py  –  Smooth Dynamic Time Warping: time-variant time-shift estimation.

Algorithm of Compton & Hale (2014) with adaptations from Cui & Margrave
(CREWES 2015) and Cui M.Sc. thesis (2015).

Port of DTWs.m from the CREWES MATLAB toolbox.
"""

import numpy as np


def DTWs(
    s1: np.ndarray,
    s2: np.ndarray,
    L: int,
    j: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate smooth time-variant time shifts between two 1-D traces.

    Compared with DTW, DTWs works on a coarse sample grid (``j``) and
    interpolates locally-optimal sub-paths rather than enforcing a strict
    block-wise constraint.  DTWs(h=1) is equivalent to DTW(b=1).

    Parameters
    ----------
    s1 : np.ndarray, 1D, length N
        Input trace to be warped.
    s2 : np.ndarray, 1D, length N
        Reference trace.
    L : int
        Half-lag range.  2*L+1 lags are searched.
    j : np.ndarray, 1D (integer indices, 0-based)
        Coarse sample positions at which locally-optimal sub-paths are
        evaluated, e.g. ``np.arange(0, N, h)``.  DTWs with h=1 is
        equivalent to DTW with b=1.

    Returns
    -------
    e : np.ndarray, shape (2*L+1, N)
        Alignment error array.
    dint : np.ndarray, shape (2*L+1, N)
        Accumulated distance array interpolated back to full sample grid.
    M : np.ndarray, shape (2*L+1, Nj-1), int
        Locally-optimal sub-path slopes at each coarse interval.
    uint : np.ndarray, 1D, length N
        Smoothly interpolated optimal lag sequence (samples).
        Estimated time shifts = uint * dt.
    """
    s1 = np.asarray(s1, dtype=float).ravel()
    s2 = np.asarray(s2, dtype=float).ravel()
    j  = np.asarray(j,  dtype=int).ravel()
    N  = len(s1)
    Nj = len(j)

    # ------------------------------------------------------------------
    # Step 1 – alignment error array  e  (2L+1 × N)
    # e[m+L, n] = |s1[n] - s2[n+m]| with zero-padding outside [0,N)
    # ------------------------------------------------------------------
    e = np.zeros((2 * L + 1, N))
    for n in range(N):
        for m in range(-L, L + 1):
            nm = n + m
            if nm < 0 or nm >= N:
                e[m + L, n] = abs(s1[n])
            else:
                e[m + L, n] = abs(s1[n] - s2[nm])

    # ------------------------------------------------------------------
    # Step 2 – accumulation on the coarse grid
    # d  : shape (2L+1, Nj)   – accumulated cost at each coarse node
    # M  : shape (2L+1, Nj-1) – locally-optimal slope q for each interval
    # ------------------------------------------------------------------
    d = np.full((2 * L + 1, Nj), np.inf)
    M = np.zeros((2 * L + 1, Nj - 1), dtype=int)

    # Initialise first coarse column with the error at j[0]
    d[:, 0] = e[:, j[0]]

    for a in range(1, Nj):
        h = j[a] - j[a - 1]               # coarse step in fine samples

        for m in range(-L, L + 1):
            row = m + L
            dj_best = np.inf
            q_best  = 0

            ql = max(-h, m - L)
            qh = min( h, m + L)

            for q in range(ql, qh + 1):
                prev_row = m - q + L       # row at previous coarse node
                if prev_row < 0 or prev_row > 2 * L:
                    continue

                dq = d[prev_row, a - 1]
                if np.isinf(dq):
                    continue

                # Accumulate error along the sub-path from j[a-1] to j[a]
                # (p steps back from j[a], i.e. sample indices j[a]-p)
                for p in range(h):
                    sample_idx = j[a] - p   # fine sample index
                    if sample_idx < 0 or sample_idx >= N:
                        continue

                    # Fractional lag at this step
                    k = p * q / h if h != 0 else 0.0
                    kr = k - int(k)

                    if kr == 0.0:
                        lag_row = m - int(k) + L
                        lag_row = max(0, min(2 * L, lag_row))
                        dq += e[lag_row, sample_idx]
                    else:
                        kl = int(np.floor(k))
                        kh = int(np.ceil(k))
                        row_l = max(0, min(2 * L, m - kl + L))
                        row_h = max(0, min(2 * L, m - kh + L))
                        el = e[row_l, sample_idx]
                        eh = e[row_h, sample_idx]
                        if q > 0:
                            dq += el * (1.0 - kr) + eh * kr
                        else:
                            dq += el * (-kr) + eh * (1.0 + kr)

                if dq < dj_best:
                    dj_best = dq
                    q_best  = q

            if not np.isinf(dj_best):
                d[row, a] = dj_best
                M[row, a - 1] = q_best

    # ------------------------------------------------------------------
    # Step 3 – interpolate d back to the full sample grid
    # ------------------------------------------------------------------
    dint = np.zeros((2 * L + 1, N))
    for a in range(Nj - 1):
        width = j[a + 1] - j[a]
        dint[:, j[a]: j[a + 1]] = d[:, a: a + 1] * np.ones((1, width))
    # Fill from the last coarse node to end of trace
    dint[:, j[-1]:] = d[:, -1: ] * np.ones((1, N - j[-1]))

    # ------------------------------------------------------------------
    # Step 4 – backtracking on the coarse grid
    # ------------------------------------------------------------------
    u_coarse = np.zeros(Nj, dtype=int)
    u_coarse[-1] = int(np.argmin(d[:, -1]))

    for a in range(Nj - 2, -1, -1):
        u_coarse[a] = u_coarse[a + 1] - M[u_coarse[a + 1], a]

    # Convert row indices to signed lag values
    u_signed = u_coarse - L

    # ------------------------------------------------------------------
    # Step 5 – linear interpolation of u back to the full sample grid
    # ------------------------------------------------------------------
    fine = np.arange(N)
    uint = np.interp(fine, j, u_signed.astype(float))

    return e, dint, M, uint
