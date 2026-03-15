"""
syncline0.py
------------
Build and display a buried syncline velocity model, replicating syncline0.m.

Run with:
    python syncline0.py
"""

import numpy as np
import matplotlib.pyplot as plt
from synclinemodel import synclinemodel

# ── Model parameters ──────────────────────────────────────────────────────────
dx        = 5       # grid spacing (m)
xmax      = 2000    # max x (m)
zmax      = 1000    # max z (m)
vhigh     = 4000    # velocity below syncline (m/s)
vlow      = 2000    # velocity above syncline (m/s)
zsyncline = 300     # depth to flat interface (m)
zfocal    = 100     # depth to focal point of syncline (m)
radius    = 500     # radius of syncline semi-circle (m)
            # NOTE: radius + zfocal (600) > zsyncline (300) → syncline exists

# ── Build model ───────────────────────────────────────────────────────────────
vel, x, z = synclinemodel(dx, xmax, zmax, vhigh, vlow,
                           zsyncline=zsyncline, radius=radius, zfocal=zfocal)

print(f"Velocity model shape : {vel.shape}  (nz={len(z)}, nx={len(x)})")
print(f"Velocity range       : {vel.min():.0f} – {vel.max():.0f} m/s")

# ── Display ───────────────────────────────────────────────────────────────────
# Mirror of MATLAB's  plotimage(vel - mean(vel(:)), z, x)
vel_dm = vel - vel.mean()   # de-meaned, highlights the contrast

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: raw velocity
im0 = axes[0].imshow(vel, extent=[x[0], x[-1], z[-1], z[0]],
                     aspect='auto', cmap='jet')
plt.colorbar(im0, ax=axes[0], label='Velocity (m/s)')
axes[0].set_title('Syncline velocity model')
axes[0].set_xlabel('X (m)')
axes[0].set_ylabel('Z (m)')

# Right: de-meaned (matches MATLAB plotimage call)
im1 = axes[1].imshow(vel_dm, extent=[x[0], x[-1], z[-1], z[0]],
                     aspect='auto', cmap='seismic')
plt.colorbar(im1, ax=axes[1], label='ΔVelocity (m/s)')
axes[1].set_title('De-meaned velocity (as in MATLAB plotimage)')
axes[1].set_xlabel('X (m)')
axes[1].set_ylabel('Z (m)')

plt.tight_layout()
plt.savefig('syncline0_model.png', dpi=150)
plt.show()
print("Figure saved to syncline0_model.png")
