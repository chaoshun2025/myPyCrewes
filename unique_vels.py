"""unique_vels.py – extract unique velocity values from a model array.

Mirrors the MATLAB ``unique_vels`` from the Margrave / CREWES seismic toolbox.
"""

import numpy as np


def unique_vels(model):
    """
    Return the unique velocity values present in *model*.

    Parameters
    ----------
    model : array_like
        Velocity model array (any shape).  The function operates on the
        flattened values.

    Returns
    -------
    vrefs : np.ndarray
        1-D array of unique velocity values found in *model*, in the order
        they are first encountered when scanning the flattened array.

    Notes
    -----
    The MATLAB original iterates over the flattened model, peeling off the
    first unique value at each step.  This Python version replicates that
    *first-occurrence* ordering via :func:`numpy.unique` with
    ``return_index=True``.
    """
    model = np.asarray(model, dtype=float).ravel()
    _, first_idx = np.unique(model, return_index=True)
    # Sort by first occurrence (matches the MATLAB loop ordering)
    vrefs = model[np.sort(first_idx)]
    return vrefs
