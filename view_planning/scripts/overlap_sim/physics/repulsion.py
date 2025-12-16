"""
Gaussian Repulsion Field (GPU-safe)
-----------------------------------
F_rep = amp * exp(-‖d‖² / (2σ²)) * (d / (‖d‖ + ε))
"""

try:
    import cupy as xp
    GPU = True
except ImportError:
    import numpy as xp
    GPU = False


def _wrap_delta(delta, grid_size):
    if grid_size is None:
        return delta
    half = grid_size / 2.0
    return xp.where(xp.abs(delta) > half,
                    delta - xp.sign(delta) * grid_size,
                    delta)


def compute_gaussian_repulsion(center, others, sigma=10.0, amp=100.0, grid_size=None):
    if others is None or len(others) == 0:
        return 0.0, 0.0

    c = xp.asarray(center, dtype=xp.float64)
    O = xp.asarray(others, dtype=xp.float64)
    diff = _wrap_delta(c[None, :] - O, grid_size)

    dx = diff[:, 0]
    dy = diff[:, 1]
    r2 = dx * dx + dy * dy
    r = xp.sqrt(r2) + 1e-12
    coeff = amp * xp.exp(-r2 / (2.0 * sigma ** 2))

    fx = xp.sum(coeff * (dx / r))
    fy = xp.sum(coeff * (dy / r))
    return fx, fy


def compute_repulsion_field(X, Y, centers, sigma=10.0, amp=100.0, grid_size=None):
    H, W = X.shape
    Fx = xp.zeros((H, W), dtype=xp.float64)
    Fy = xp.zeros((H, W), dtype=xp.float64)
    if centers is None or len(centers) == 0:
        return Fx, Fy

    C = xp.asarray(centers, dtype=xp.float64)
    cx = C[:, 0][:, None, None]
    cy = C[:, 1][:, None, None]
    Xg = X[None, :, :]
    Yg = Y[None, :, :]

    dx = _wrap_delta(Xg - cx, grid_size)
    dy = _wrap_delta(Yg - cy, grid_size)
    r2 = dx * dx + dy * dy
    r = xp.sqrt(r2) + 1e-12

    coeff = amp * xp.exp(-r2 / (2.0 * sigma ** 2))
    Fx = (coeff * (dx / r)).sum(axis=0)
    Fy = (coeff * (dy / r)).sum(axis=0)
    return Fx, Fy
