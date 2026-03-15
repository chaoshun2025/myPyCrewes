"""
DTW.py  –  Dynamic Time Warping: time-variant time-shift estimation.

Algorithm of Hale (2013) with adaptations from Cui & Margrave (CREWES 2014)
and Cui M.Sc. thesis (2015).

Port of DTW.m from the CREWES MATLAB toolbox.
"""

import numpy as np


def DTW(
    s1: np.ndarray,
    s2: np.ndarray,
    L: int,
    b: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate time-variant time shifts between two 1-D traces via
    Dynamic Time Warping.

    Parameters
    ----------
    s1 : np.ndarray, 1D, length N
        Input trace to be warped.
    s2 : np.ndarray, 1D, length N
        Reference trace.  Time shifts are computed relative to s2.
    L : int
        Half-lag range.  2*L+1 lags are searched.
    b : int
        Smoothness constraint: the lag sequence is allowed to change by at
        most 1 sample per block of b samples.

    Returns
    -------
    e : np.ndarray, shape (2*L+1, N)
        Alignment error array.  e[m+L, n] = |s1[n] - s2[n+m]|.
    d : np.ndarray, shape (2*L+1, N)
        Accumulated distance (cost) array from dynamic programming.
    u : np.ndarray, 1D, length N
        Optimal lag sequence in samples.
        Estimated time shifts = u * dt  (dt = time sample interval).
    """
    s1 = np.asarray(s1, dtype=float).ravel()
    s2 = np.asarray(s2, dtype=float).ravel()
    N = len(s1)

    # ------------------------------------------------------------------
    # Step 1 – build alignment error array  e  (shape: (2L+1, N))
    # Row index  m+L  corresponds to lag  m  in  [-L, L].
    # e[m+L, n] = |s1[n] - s2[clip(n+m)]|
    # ------------------------------------------------------------------
    e = np.zeros((2 * L + 1, N))
    for n in range(N):                           # 0-based column
        for m in range(-L, L + 1):              # lag
            npm = max(0, min(n + m, N - 1))     # clipped index into s2
            e[m + L, n] = abs(s1[n] - s2[npm])

    # ------------------------------------------------------------------
    # Step 2 – dynamic programming: build accumulated distance array d
    # ------------------------------------------------------------------
    d = np.zeros_like(e)

    for n in range(N):                           # 0-based
        n1 = max(0, min(N - 1, n - 1))          # n-1, clamped
        nb = max(0, min(N - 1, n - b))           # n-b, clamped

        for m in range(-L, L + 1):
            row = m + L                          # 0-based row index
            mm = max(row - 1, 0)                 # row for lag m-1
            mp = min(row + 1, 2 * L)             # row for lag m+1

            dm = d[mm, nb]
            dn = d[row, n1]
            dp = d[mp, nb]

            # accumulate error along the two diagonal paths (nb → n1)
            for kb in range(nb + 1, n1 + 1):    # nb+1 .. n1 inclusive
                dm += e[mm, kb]
                dp += e[mp, kb]

            d[row, n] = e[row, n] + min(dm, dn, dp)

    # ------------------------------------------------------------------
    # Step 3 – backtracking: recover the optimal lag sequence u
    # ------------------------------------------------------------------
    u = np.zeros(N, dtype=int)
    u[-1] = int(np.argmin(d[:, -1]))
    m = u[-1]                                    # current row index (0-based)
    n = N - 1                                    # current column (0-based)

    while n != 0:
        n1 = max(0, min(N - 1, n - 1))
        nb = max(0, min(N - 1, n - b))
        mm = max(m - 1, 0)
        mp = min(m + 1, 2 * L)

        dm = d[mm, nb]
        di = d[m,  n1]
        dp = d[mp, nb]

        for kb in range(nb + 1, n1 + 1):
            dm += e[mm, kb]
            dp += e[mp, kb]

        ind = int(np.argmin([dm, di, dp]))       # 0=left, 1=ident, 2=right

        prev_m = m
        if ind == 0:
            m = mm
        elif ind == 2:
            m = mp

        n -= 1
        u[n] = m

        # If we moved diagonally, fill the block nb..n1-1 with the same lag
        if m == mm or m == mp:
            for kb in range(nb, n1):             # nb .. n1-1
                n -= 1
                if n >= 0:
                    u[n] = m

    # Convert row indices to signed lag values
    u = u - L

    return e, d, u
