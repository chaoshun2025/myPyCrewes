import numpy as np


def interpbl(t, s, tint, n=8, m=2):
    """
    INTERPBL: band-limited sinc function interpolation.

    sint = interpbl(t, s, tint, n, m)

    Reconstructs the continuous band-limited signal from its samples using
    a truncated, Gaussian-tapered sinc interpolation kernel.

    Parameters
    ----------
    t : array_like, shape (nt,)
        Regularly-spaced time coordinates of the input samples.
    s : array_like, shape (nt,) or (nt, ntraces)
        Input samples.  If 2-D, interpolation is applied to every column
        (trace) with the same ``tint`` sites.
    tint : array_like, shape (nint,)
        Times at which interpolated values are desired.  NaN entries are
        allowed; they will produce 0 in the output.
    n : int, optional
        Half-length of the sinc function.  n=4 ~ spline quality; n=8 is
        better.  Default = 8.
    m : float, optional
        The Gaussian taper is ``m`` standard deviations down at the sinc
        truncation point.  Default = 2.

    Returns
    -------
    sint : np.ndarray, shape (nint, ntraces)
        Interpolated samples.  Single-trace input returns shape (nint, 1);
        reshape as needed.

    Notes
    -----
    Translated from the CREWES MATLAB ``interpbl2`` / ``interpbl`` by
    G.F. Margrave.
    """
    t = np.asarray(t, dtype=float).ravel()
    tint_orig = None

    # ---- handle NaNs in tint ----
    tint = np.asarray(tint, dtype=float).ravel()
    nan_mask = np.isnan(tint)
    if np.any(nan_mask):
        tint_orig = tint.copy()
        tint = tint[~nan_mask]

    nt = len(t)

    if np.sum(np.abs(np.diff(np.diff(t)))) > 1e4 * np.finfo(float).eps:
        raise ValueError("input times must fall on a regular grid")

    # ---- shift so t starts at 0 ----
    t1 = t[0]
    t = t - t1
    tint = tint - t1

    dt = t[1] - t[0]
    tmin, tmax = t.min(), t.max()

    # ---- normalise s to 2-D (nt x ntraces) ----
    s = np.asarray(s, dtype=float)
    if s.ndim == 1:
        s = s[:, np.newaxis]
        nr, nc = s.shape
        ntr = nc
    else:
        nr, nc = s.shape
        if nr == nt:
            ntr = nc
        elif nc == nt:
            s = s.T
            ntr = nr
        else:
            raise ValueError("input variables t and s have incompatible sizes")

    # ---- pad s with n zeros top and bottom ----
    s2 = np.vstack([np.zeros((n, ntr)), s, np.zeros((n, ntr))])
    t2 = np.arange(s2.shape[0]) * dt - n * dt   # time axis of padded signal

    # ---- build sinc table ----
    inc = 50
    dt2 = dt / inc
    tsinc = np.arange(-n * dt, n * dt + dt2 / 2, dt2)
    one_over_sigma = m / tsinc[-1]
    sink = (np.sinc(tsinc / dt) *
            np.exp(-(one_over_sigma * tsinc) ** 2))

    nint = len(tint)
    sint = np.zeros((nint, ntr))
    small = 100 * np.finfo(float).eps

    for k in range(nint):
        tk = tint[k]
        if tk < tmin or tk > tmax:
            continue

        # fractional sample number in s2 (1-based in MATLAB -> 0-based here)
        kint = (tk - t2[0]) / dt   # 0-based fractional index

        if abs(kint - round(kint)) < small:
            sint[k, :] = s2[int(round(kint)), :]
        else:
            kl = int(np.floor(kint)) + np.arange(-n + 1, 1)   # n left indices
            kr = int(np.ceil(kint)) + np.arange(0, n)          # n right indices

            # corresponding look-up indices in the sinc table (0-based)
            klsinc = (kl * inc - round(tk / dt2)).astype(int)
            krsinc = (kr * inc - round(tk / dt2)).astype(int)

            all_idx = np.concatenate([kl, kr])
            all_sinc_idx = np.concatenate([klsinc, krsinc])

            op = sink[all_sinc_idx]
            sint[k, :] = np.dot(op, s2[all_idx, :])

    # ---- re-insert NaN positions ----
    if tint_orig is not None:
        sint2 = np.zeros((len(tint_orig), ntr))
        sint2[~nan_mask, :] = sint
        sint = sint2

    return sint
