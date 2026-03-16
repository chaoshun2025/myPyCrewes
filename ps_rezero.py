"""ps_rezero.py – re-zero the temporal zero-pad in phase-shift extrapolation.

Mirrors the MATLAB ``ps_rezero`` from the Ferguson / Margrave CREWES toolbox.
"""

import numpy as np
from near import near
from fktran import fktran
from ifktran import ifktran


def ps_rezero(fdata, f, dx, tmax):
    """
    Re-zero the temporal zero-pad in a phase-shift extrapolated f-k dataset.

    After several downward-continuation steps, energy leaks into the
    zero-padded portion of the time axis due to operator wrap-around.
    This function transforms back to the t-x domain, zeros all samples
    beyond *tmax*, and transforms forward again.

    Parameters
    ----------
    fdata : np.ndarray (complex), shape (nf, nx)
        f-k spectrum with the k axis **wrapped** (natural FFT ordering) and
        only positive frequencies included.
    f : array_like, length nf
        Frequency coordinate vector (Hz) for ``fdata``.
    dx : float
        Spatial sample interval (m).
    tmax : float
        Time (s) beyond which the data are set to zero.

    Returns
    -------
    fdataout : np.ndarray (complex), shape (nf, nx)
        f-k spectrum with the zero-pad re-zeroed, same frequency indices
        as the input.

    Notes
    -----
    The function determines the Nyquist frequency from *f* and finds the
    smallest ``dt`` (starting at 8 ms and halving) such that
    ``0.5 / dt >= fmax``.  It then builds a full frequency axis up to
    ``fnyq``, embeds ``fdata`` at the appropriate indices, calls
    :func:`ifktran`, zeros samples beyond ``tmax``, and calls :func:`fktran`
    to return to the f-k domain.  Only the frequency rows corresponding to
    the original ``f`` range are returned.
    """
    fdata = np.asarray(fdata, dtype=complex)
    f     = np.asarray(f, dtype=float).ravel()
    nf_in, nc = fdata.shape

    # ------------------------------------------------------------------
    # Determine dt such that fnyq >= fmax
    # ------------------------------------------------------------------
    df   = f[1] - f[0]
    fmax = f[-1]
    dt   = 0.008
    fnyq = 0.5 / dt
    while fnyq < fmax:
        dt   /= 2.0
        fnyq  = 0.5 / dt

    # True maximum time in the padded record (with fiddle matching MATLAB)
    Tmax = 1.0 / df - dt          # "necessary fiddle"

    # ------------------------------------------------------------------
    # Build full f and kx axes
    # ------------------------------------------------------------------
    fnew = np.arange(0.0, fnyq + df * 0.5, df)   # 0 … fnyq in steps of df
    nf_new = len(fnew)

    # Wrapped kx axis: fftshift of [-nc … nc-2] / (2*nc*dx) (MATLAB convention)
    kx = np.fft.fftshift(
        np.arange(-nc, nc, 2, dtype=float) / (2.0 * nc * dx)
    )

    # Embed fdata into a full-bandwidth spectrum
    indf = near(fnew, f[0], f[-1])
    data = np.zeros((nf_new, nc), dtype=complex)
    data[indf, :] = fdata
    data[-1, :]   = 0.0           # zero the Nyquist row (real)

    # ------------------------------------------------------------------
    # Inverse f-k → t-x
    # ------------------------------------------------------------------
    tdata, t, x = ifktran(data, fnew, kx, 0, 0)

    # ------------------------------------------------------------------
    # Zero the temporal zero-pad (samples beyond tmax … Tmax)
    # ------------------------------------------------------------------
    indt = near(t, tmax, Tmax)
    tdata[indt, :] = 0.0

    # ------------------------------------------------------------------
    # Forward t-x → f-k
    # ------------------------------------------------------------------
    data_out, _, _ = fktran(tdata, t, x, 0, 0, 0.0, 0)

    # Return only the rows that correspond to the original f range
    fdataout = data_out[indf, :]
    return fdataout
