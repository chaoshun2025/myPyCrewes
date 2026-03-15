import numpy as np


def mwindow(n, percent=10):
    """
    Returns the N-point Margrave window as a 1-D NumPy array.

    The window is a boxcar over the central (100 - 2*percent)% of samples,
    with a raised-cosine (Hanning-style) taper on each end.

    Parameters
    ----------
    n : int or array-like
        Length of the window.  If an array is supplied, len(n) is used.
    percent : float, optional
        Percentage taper on each end of the window.  Default is 10.

    Returns
    -------
    w : np.ndarray, shape (n,)
        The Margrave window.

    Raises
    ------
    ValueError
        If percent is outside the valid range [0, 50].
    """
    # Accept vectors just like the MATLAB version
    if np.ndim(n) > 0:
        n = len(n)
    n = int(n)

    if percent > 50 or percent < 0:
        raise ValueError("invalid percent for mwindow")

    # Number of taper samples (even, matching MATLAB's 2*floor(m/2))
    m = int(2 * np.floor(percent * n / 100.0))   # always even

    # MATLAB's hanning(m) is the *periodic* Hanning window:
    #   h[k] = 0.5 - 0.5 * cos(2*pi*k / m),  k = 0, 1, ..., m-1
    # Note: scipy.signal.hann / np.hanning are *symmetric* (period m-1),
    # so we build the periodic version explicitly.
    if m > 0:
        k = np.arange(m)
        h = 0.5 - 0.5 * np.cos(2.0 * np.pi * k / m)

        # MATLAB indices h(1:m/2) are 1-based → Python [0 : m//2]
        # MATLAB indices h(m/2:-1:1) reverse → Python [m//2-1 : -1 : -1]
        half = m // 2
        ramp_up   = h[0      : half]       # h(1:m/2)      in MATLAB 1-based
        ramp_down = h[half-1 :: -1]       # h(m/2:-1:1)   in MATLAB 1-based
    else:
        ramp_up   = np.array([])
        ramp_down = np.array([])

    plateau = np.ones(n - m)
    w = np.concatenate([ramp_up, plateau, ramp_down])
    return w


# ---------------------------------------------------------------------------
# Quick self-test (mirrors the MATLAB behaviour)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    for n, pct in [(100, 10), (101, 10), (100, 20), (50, 0), (64, 25)]:
        w = mwindow(n, pct)
        assert len(w) == n, f"length mismatch: got {len(w)}, expected {n}"
        print(f"mwindow({n:3d}, {pct:2d}%): len={len(w)}, "
              f"min={w.min():.4f}, max={w.max():.4f}, "
              f"first5={np.round(w[:5], 4)}")

    # Plot a couple of examples
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, (n, pct) in zip(axes, [(100, 10), (128, 20)]):
        ax.plot(mwindow(n, pct))
        ax.set_title(f"mwindow({n}, {pct}%)")
        ax.set_xlabel("sample")
        ax.set_ylabel("amplitude")
        ax.grid(True)
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/mwindow_test.png", dpi=120)
    plt.show()
    print("Plot saved.")
