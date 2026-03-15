import numpy as np


def reflec(
    tmax: float,
    dt: float,
    amp: float = 0.2,
    m: int = 3,
    n: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    REFLEC – Synthetic pseudo-random reflectivity.

    Generates a spiky reflectivity series by drawing Gaussian noise and
    raising each sample to the power *m* (preserving sign for odd m).

    Parameters
    ----------
    tmax : float
        Record length (seconds).
    dt : float
        Sample interval (seconds).
    amp : float, optional
        Maximum reflection coefficient magnitude.  Default = 0.2.
    m : int, optional
        Exponentiation power.  Should be odd to preserve sample sign.
        Default = 3.
    n : float or None, optional
        Random-number seed.  If None, a clock-based seed is used.

    Returns
    -------
    r : np.ndarray, 1D
        Reflectivity series.
    t : np.ndarray, 1D
        Time coordinate vector from 0 to tmax (inclusive).
    """
    m = int(round(m))

    t = np.arange(0.0, tmax + dt / 2, dt)   # inclusive of tmax

    # Seed the RNG (match MATLAB behaviour: abs of seed)
    if n is None:
        seed = int(np.floor(np.random.default_rng().integers(0, 2**31)))
    else:
        seed = int(abs(n))
    rng = np.random.default_rng(seed)

    noise = rng.standard_normal(len(t))

    # Raise to power m, preserving sign for even exponents
    if m % 2 == 0:
        r = (noise ** m) * np.sign(noise)
    else:
        r = noise ** m

    # Normalise and scale
    max_abs = np.max(np.abs(r))
    if max_abs > 0:
        r = amp * r / max_abs

    return r, t
