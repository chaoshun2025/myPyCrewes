"""filtf.py – apply a bandpass filter to a trace (or gather).

Direct port of Margrave/CREWES ``filtf.m`` (G.F. Margrave, May 1991;
modified H.D. Geiger, Feb 2003).
"""

import numpy as np
from gauss import gauss
from fftrl import fftrl
from ifftrl import ifftrl
from mwindow import mwindow
from hilbm import hilbm


def _padpow2(s):
    """Zero-pad *s* along axis-0 to the next power of 2."""
    n = s.shape[0]
    n2 = 1 << int(np.ceil(np.log2(n))) if n > 1 else 1
    if n2 == n:
        return s
    pad = [(0, n2 - n)] + [(0, 0)] * (s.ndim - 1)
    return np.pad(s, pad)


def filtf(trin, t, fmin, fmax, phase=0, max_atten=80.0):
    """
    Bandpass-filter a trace (or gather) in the frequency domain.

    Filter slopes are Gaussian-shaped.  The trace is automatically padded
    to the next power of 2; the pad is removed before returning.

    Parameters
    ----------
    trin : array-like, 1-D or 2-D
        Input trace(s).  For 2-D arrays traces are in **columns** (first
        axis is time).
    t : array-like
        Time coordinate vector matching the first axis of *trin*.
    fmin : array-like, length 1 or 2
        Low-cut (high-pass) edge.
        ``fmin[0]`` – 3 dB-down frequency (Hz); set to 0 for low-pass only.
        ``fmin[1]`` – Gaussian width (Hz); default 5 Hz.
    fmax : array-like, length 1 or 2
        High-cut (low-pass) edge.
        ``fmax[0]`` – 3 dB-down frequency (Hz); set to 0 for high-pass only.
        ``fmax[1]`` – Gaussian width (Hz); default 10 % of Fnyquist.
    phase : {0, 1}, optional
        0 → zero-phase (default).  1 → minimum-phase (approximate).
    max_atten : float, optional
        Maximum attenuation in dB.  Default 80.

    Returns
    -------
    trout : np.ndarray
        Filtered trace(s), same shape as *trin*.
    """
    trin = np.asarray(trin, dtype=float)
    t    = np.asarray(t,    dtype=float).ravel()
    dt   = t[1] - t[0]
    fnyq = 0.5 / dt

    # ---- default widths (mirrors MATLAB nargin / length checks) ----
    fmin = list(np.atleast_1d(np.asarray(fmin, dtype=float)))
    fmax = list(np.atleast_1d(np.asarray(fmax, dtype=float)))
    if len(fmin) < 2:
        fmin.append(5.0)
    if len(fmax) < 2:
        fmax.append(0.1 * fnyq)          # MATLAB: .1/(2*dt) = .1*fnyq

    # ---- normalise to column-major 2-D (nt × ntraces) ----
    nt = len(t)
    squeeze = False
    trflag  = False
    if trin.ndim == 1:
        trin    = trin[:, np.newaxis]
        squeeze = True
    else:
        if trin.shape[0] != nt and trin.shape[1] == nt:
            trin   = trin.T
            trflag = True

    nt = trin.shape[0]
    ntr = trin.shape[1]

    dbd = 3.0   # dB-down level for fmin/fmax edge definition

    # ---- remove DC (MATLAB: trinDC = ones(nt,1)*sum(trin)/nt) ----
    # sum(trin)/nt == mean per column; ones(nt,1)* broadcasts to (nt, ntr)
    trinDC = np.mean(trin, axis=0, keepdims=True) * np.ones((nt, 1))
    trin   = trin - trinDC

    # ---- pad to next power of 2, FFT ----
    trin_pad = _padpow2(trin)
    npad     = trin_pad.shape[0]
    t_pad    = dt * np.arange(npad)
    Trin, f  = fftrl(trin_pad, t_pad)
    nf       = len(f)
    df       = f[1] - f[0]

    gnot = 10.0 ** (-abs(max_atten) / 20.0)

    # ---- low-cut (high-pass) Gaussian ----
    if fmin[0] > 0:
        fnotl = fmin[0] + np.sqrt(np.log(10) * dbd / 20.0) * fmin[1]
        glow  = gnot + gauss(f, fnotl, fmin[1])   # shape (nf,)
        if phase != 1:
            glow[0] = 0.0       # force DC to zero for zero-phase
    else:
        glow  = None
        fnotl = 0.0

    # ---- high-cut (low-pass) Gaussian ----
    if fmax[0] > 0:
        fnoth = fmax[0] - np.sqrt(np.log(10) * dbd / 20.0) * fmax[1]
        ghigh = gnot + gauss(f, fnoth, fmax[1])   # shape (nf,)
    else:
        ghigh = None
        fnoth = 0.0

    # ---- assemble filter (exact translation of MATLAB splice logic) ----
    #
    # MATLAB uses 1-based indexing.  Key identities:
    #   MATLAB a(1:k)      → Python a[0:k]      (k elements)
    #   MATLAB a(k+1:end)  → Python a[k:nf]     (nf-k elements)
    #
    # low-pass only  (nl == 0):
    #   MATLAB: [fltr(1:nh);   ghigh(nh+1:end)]
    #   Python: [fltr[0:nh],   ghigh[nh:nf]]    → nf elements ✓
    #
    # high-pass only (nh == 0):
    #   MATLAB: [glow(1:nl+1); fltr(nl+2:end)]
    #   Python: [glow[0:nl+1], fltr[nl+1:nf]]   → nf elements ✓
    #
    # band-pass:
    #   MATLAB: [glow(1:nl+1); fltr(nl+2:nf)] .* [fltr(1:nh); ghigh(nh+1:end)]
    #   Python: [glow[0:nl+1], fltr[nl+1:nf]] *  [fltr[0:nh], ghigh[nh:nf]]
    #
    # NOTE: the old Python port had `fltr[nl+2:]` which skips index nl+1
    #       and produces a vector of length nf-1, causing the broadcast error.
    #       The correct translation of MATLAB's `nl+2:end` (1-based) is
    #       Python's `nl+1:` (0-based), preserving all nf elements.

    fltr = np.ones(nf)
    nl   = int(np.floor(fnotl / df)) if fnotl > 0 else 0
    nh   = int(np.ceil(fnoth  / df)) if fnoth > 0 else 0

    if nl == 0 and ghigh is not None:
        # low-pass only
        fltr = np.concatenate([fltr[:nh], ghigh[nh:]])

    elif nh == 0 and glow is not None:
        # high-pass only
        fltr = np.concatenate([glow[:nl + 1], fltr[nl + 1:]])

    elif glow is not None and ghigh is not None:
        # band-pass: element-wise product of the two spliced halves
        left  = np.concatenate([glow[:nl + 1],  fltr[nl + 1:]])   # high-pass shape
        right = np.concatenate([fltr[:nh],       ghigh[nh:]])      # low-pass  shape
        fltr  = left * right
        fltr  = fltr / np.max(np.abs(fltr))
    # else: both fmin[0]==0 and fmax[0]==0 → all-pass, fltr stays ones

    # ---- make minimum phase if required ----
    if phase == 1:
        # Build conjugate-symmetric two-sided spectrum of length npad = 2*(nf-1)
        # MATLAB: L1=1:length(fltr); L2=length(fltr)-1:-1:2  (1-based)
        #   → Python L1 = 0..nf-1, L2 = nf-2..1  (0-based, reversed)
        L2      = np.arange(nf - 2, 0, -1)         # nf-2 elements
        symspec = np.concatenate([fltr, np.conj(fltr[L2])])  # length 2*(nf-1) = npad
        cmpx    = np.log(symspec) + 1j * np.zeros(len(symspec))
        fltr    = np.exp(np.conj(hilbm(cmpx)))
        fltr    = fltr[:nf]     # keep only positive-frequency half

    # ---- apply filter ----
    # MATLAB: Trin .* (fltr(1:length(f)) * ones(1, ntraces))
    fltr_col = np.asarray(fltr[:nf])[:, np.newaxis]   # (nf, 1)
    if Trin.ndim == 1:
        Trin = Trin[:, np.newaxis]
    trout, _ = ifftrl(Trin * fltr_col, f)              # (npad, ntr) real

    # ---- truncate to original length, apply taper ----
    # MATLAB: if size(trout,1)~=nt; win=mwindow(nt,10); trout=trout(1:nt,:).*win
    if trout.shape[0] != nt:
        win   = mwindow(nt, 10)
        trout = trout[:nt, :] * win[:, np.newaxis]

    # ---- restore DC ----
    # MATLAB: trinDC = trinDC * abs(fltr(1))
    #         troutDC = ones(nt,1)*sum(trout)/nt
    #         trout = trout - troutDC + trinDC
    trinDC  = trinDC  * float(np.abs(fltr[0]))
    troutDC = np.mean(trout, axis=0, keepdims=True) * np.ones((nt, 1))
    trout   = trout - troutDC + trinDC

    # ---- restore original orientation / shape ----
    if trflag:
        trout = trout.T
    if squeeze:
        trout = trout[:, 0]

    return trout
