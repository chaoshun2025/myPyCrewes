import numpy as np


def near(v, val1, val2=None):
    """
    NEAR: find indices of the closest samples to one or two values.

    L = near(v, val1, val2)

    Searches the vector v for the index L1 whose entry is closest to val1
    and L2 whose entry is closest to val2, then returns the contiguous
    index range between them.

    Parameters
    ----------
    v : array_like
        Input vector (may contain NaN values; NaNs are ignored when
        searching for val1, but *not* when searching for val2 — matching
        the MATLAB original exactly).
    val1 : float
        First search value.
    val2 : float, optional
        Second search value.  Default = val1 (returns scalar-length range).

    Returns
    -------
    L : np.ndarray of int (0-based)
        When L1 <= L2: ``np.arange(min(L1), max(L2) + 1)``
        When L1 >  L2: ``np.arange(max(L1), min(L2) - 1, -1)``

    Notes
    -----
    * L1 is found using only non-NaN entries of v (via ``ilive``).
    * L2 is found from the full v, including NaNs — this mirrors the
      MATLAB code exactly (``test=abs(v-val2); L2=find(test==min(test))``).
    * When multiple indices are equidistant from val1 or val2, all ties
      are kept and ``min``/``max`` resolves the range endpoints, consistent
      with the MATLAB ``min(L1):max(L2)`` / ``max(L1):-1:min(L2)`` idiom.
    """
    v = np.asarray(v, dtype=float).ravel()

    if val2 is None:
        val2 = val1

    # --- L1: search only among non-NaN entries ---
    ilive = np.where(~np.isnan(v))[0]          # 0-based live indices
    test1 = np.abs(v[ilive] - val1)
    L1 = ilive[test1 == test1.min()]            # all tied winners (0-based)

    # --- L2: search full v (NaN - val2 = NaN, so NaN entries never win) ---
    test2 = np.abs(v - val2)
    L2 = np.where(test2 == np.nanmin(test2))[0]  # all tied winners (0-based)

    # --- build range using min/max of the tie sets, exactly as MATLAB ---
    if L1.min() <= L2.max():          # mirrors:  if L1 <= L2
        L = np.arange(L1.min(), L2.max() + 1)
    else:                             # mirrors:  L=max(L1):-1:min(L2)
        L = np.arange(L1.max(), L2.min() - 1, -1)

    return L
