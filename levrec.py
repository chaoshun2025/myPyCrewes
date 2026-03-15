"""levrec.py – Levinson recursion solver for symmetric Toeplitz systems.

Direct port of Margrave toolbox ``levrec.m``.

Reference: Golub & Van Loan, *Matrix Computations*, 2nd ed., Johns Hopkins
University Press, 1989, p. 187.
"""

import numpy as np


def levrec(aa: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the symmetric Toeplitz system T·x = b via Levinson recursion.

    The Toeplitz matrix T is fully specified by the autocorrelation vector
    *aa*::

        T[i, j] = aa[|i - j|]   (0-based indices)

    Parameters
    ----------
    aa : array-like, 1-D
        Autocorrelation vector of length ≥ ``len(b)``.
        If ``aa[0] != 1`` the vector is normalised by ``max(aa)``.
    b : array-like, 1-D
        Right-hand side vector.  Converted to a column (1-D) array
        internally.

    Returns
    -------
    x : np.ndarray, 1-D
        Solution vector, same length as *b*.

    Raises
    ------
    ValueError
        If the zero-lag value is not the maximum of *aa* after normalisation
        (i.e. the supplied vector is not a valid autocorrelation).

    Notes
    -----
    This is a verbatim translation of the MATLAB Levinson recursion loop.
    All index arithmetic mirrors the original 1-based MATLAB code with a
    systematic −1 offset.
    """
    aa = np.asarray(aa, dtype=float).ravel()
    b  = np.asarray(b,  dtype=float).ravel()

    # Normalise autocorrelation so that aa[0] == 1
    if aa[0] != 1.0:
        aa = aa / np.max(aa)

    if aa[0] != np.max(aa):
        raise ValueError("Invalid autocorrelation: zero-lag is not the maximum.")

    # MATLAB:  a = aa(2:end)   →  0-based: aa[1:]
    a = aa[1:]

    n = len(b)

    # Working arrays (0-based, length n-1 like 'a')
    y = np.zeros(n - 1)   # prediction-error filter coefficients
    z = np.zeros(n - 1)   # scratch
    x = np.zeros(n)       # solution

    # Initialise  (mirrors MATLAB's k=0 state before the loop)
    # MATLAB: y(1)=-a(1); x(1)=b(1); beta=1; alpha=-a(1)
    y[0]  = -a[0]
    x[0]  =  b[0]
    beta  =  1.0
    alpha = -a[0]

    # Main Levinson recursion  (MATLAB loop: for k = 1 : n-1)
    for k in range(1, n):           # k is 1-based in MATLAB, 0-based offset here
        # MATLAB: beta = (1 - alpha^2) * beta
        beta = (1.0 - alpha ** 2) * beta

        # MATLAB: mu = (b(k+1) - a(1:k)' * x(k:-1:1)) / beta
        #   a(1:k)    → a[0:k]   (length k)
        #   x(k:-1:1) → x[k-1::-1] (x[0..k-1] reversed)
        mu = (b[k] - a[:k] @ x[:k][::-1]) / beta

        # MATLAB: nu(1:k) = x(1:k) + mu * y(k:-1:1)
        #         x(1:k)  = nu(1:k)
        #         x(k+1)  = mu
        x[:k] = x[:k] + mu * y[:k][::-1]
        x[k]  = mu

        # Update y for the next iteration (only needed when k < n-1)
        if k < n - 1:
            # MATLAB: alpha = -(a(k+1) + a(1:k)' * y(k:-1:1)) / beta
            alpha = -(a[k] + a[:k] @ y[:k][::-1]) / beta

            # MATLAB: z(1:k) = y(1:k) + alpha * y(k:-1:1)
            #         y(1:k) = z(1:k)
            #         y(k+1) = alpha
            z[:k] = y[:k] + alpha * y[:k][::-1]
            y[:k] = z[:k]
            y[k]  = alpha

    return x
