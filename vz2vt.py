"""vz2vt.py – convert V(x,z) depth model to V(x,t) and Vrms(x,t).

Mirrors the MATLAB ``vz2vt`` from the Margrave / CREWES seismic toolbox.
"""

import time
import numpy as np
from vint2t import vint2t
from vint2vrms import vint2vrms


def vz2vt(vel, seis, samprates):
    """
    Compute V(x,t) and Vrms(x,t) from a depth velocity model V(x,z).

    Takes a depth velocity model and a ZOS (zero-offset section) exploding-
    reflector model and converts them to velocity matrices in two-way
    travel-time.

    Parameters
    ----------
    vel : np.ndarray, shape (nz, nx)
        Interval velocity model in depth.  Each row is a constant depth.
    seis : np.ndarray, shape (nt, nx)
        Exploding-reflector model (ZOS).  Provides the time-axis size.
        Must have the same number of columns (traces) as *vel*.
    samprates : array_like, length 3
        ``[dz, dx, dt]`` – depth sample interval (m), lateral sample
        interval (m), and time sample interval (s) of the ZOS model.

    Returns
    -------
    vmat : np.ndarray, shape (nt, nx)
        Instantaneous velocity in two-way time  V(x,t).
    vrmsmat : np.ndarray, shape (nt, nx)
        RMS velocity in two-way time  Vrms(x,t).
    tmat : np.ndarray, shape (nz+1, nx)
        Two-way vertical travel-time matrix.  Each column holds the
        cumulative travel-times from the surface to every depth sample.
    xsamp : np.ndarray
        Lateral position vector (m).
    tsamp : np.ndarray
        Time sample vector (s) for the output matrices.

    Notes
    -----
    * The background fill velocity (for time samples deeper than the model)
      is 2800 m/s, matching the original MATLAB code.
    * Progress messages are printed to stdout every 50 traces, matching
      MATLAB's ``disp`` calls.
    """
    samprates = np.asarray(samprates, dtype=float).ravel()
    if len(samprates) < 3:
        raise ValueError(
            "samprates must be a 3-element vector [dz, dx, dt]"
        )
    dz, dx, dt = samprates[0], samprates[1], samprates[2]

    vel = np.asarray(vel, dtype=float)
    seis = np.asarray(seis, dtype=float)

    sz, sx = seis.shape   # time samples × traces
    vz, vx = vel.shape    # depth samples × traces

    if vx != sx:
        raise ValueError(
            "Exploding-reflector model 'seis' must have the same number of "
            "columns as the velocity model 'vel'"
        )

    t_start = time.time()

    # ------------------------------------------------------------------
    # Step 1 – compute two-way vertical travel-time matrix
    # ------------------------------------------------------------------
    z = dz * np.arange(vz + 1)   # length vz+1  (surface to bottom)
    xsamp = dx * np.arange(vx)
    tmat = np.zeros((vz + 1, vx))

    print("Calculating Two-way vertical travel times")

    for s in range(vx):
        # vint2t with z of length vz+1 and vint of length vz
        t1, _ = vint2t(vel[:, s], z)
        tmat[:, s] = t1

    tmat *= 2.0   # one-way → two-way

    elapsed = time.time() - t_start
    print(f"Two-way vertical travel times calculated in {elapsed:.2f} s")

    # ------------------------------------------------------------------
    # Step 2 – resample velocity matrix from depth to time
    # ------------------------------------------------------------------
    t_resamp = time.time()
    print("Resampling the velocity matrix from depth to time")

    vmat = np.full((sz, sx), 2800.0)   # background fill
    tsamp = dt * np.arange(sz)

    for s in range(vx):
        vsamp = np.empty(sz)
        vsamp[0] = vel[0, s]
        k = 1   # pointer into depth layers (0-based: layer index)
        for p in range(1, sz):
            if k <= vz - 1:            # still inside the model
                if tsamp[p] <= tmat[k, s]:
                    vsamp[p] = vel[k - 1, s]
                else:
                    k += 1
                    vsamp[p] = vel[k - 1, s]
            else:
                # Beyond the bottom of the model.
                # MATLAB: vsamp(p)=vel(k-2,s) where k is 1-based k=vz+1,
                # so vel(vz-1,s)[1-based] = vel[vz-2][0-based].
                # k here is 0-based k=vz, so vel[k-2, s] = vel[vz-2, s].
                vsamp[p] = vel[k - 2, s]

        vmat[:len(vsamp), s] = vsamp

        if (s + 1) % 50 == 0:
            elapsed = time.time() - t_resamp
            print(
                f"Column {s+1} of {sx} resampled to time in "
                f"{elapsed:.2f} s"
            )

    print("Resampling completed, now calculating RMS velocities")

    # ------------------------------------------------------------------
    # Step 3 – compute RMS velocity matrix
    # ------------------------------------------------------------------
    t_rms = time.time()
    vrmsmat = np.zeros((sz, vx))

    for s in range(vx):
        vrms = vint2vrms(vmat[:, s], tsamp)
        vrmsmat[:, s] = vrms

        if (s + 1) % 50 == 0:
            elapsed = time.time() - t_rms
            print(
                f"RMS velocities calculated for Column {s+1} of {vx} "
                f"in {elapsed:.2f} s"
            )

    total = time.time() - t_start
    print(f"V(x,t) and VRMS(x,t) calculated in {total:.2f} s")

    return vmat, vrmsmat, tmat, xsamp, tsamp
