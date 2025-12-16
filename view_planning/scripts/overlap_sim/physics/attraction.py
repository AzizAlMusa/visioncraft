"""
Attraction potential field (GPU-safe)
-------------------------------------
Implements Φ(r) = k_attr * log(r + ε) weighted by need_map.
All operations run on CuPy if available, else NumPy fallback.
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


def compute_log_potential(X, Y, viewpoints, need_map,
                          k_attr=10.0, eps=1e-9, grid_size=None):
    H, W = X.shape
    if viewpoints is None or len(viewpoints) == 0:
        Z = xp.zeros((H, W), dtype=xp.float64)
        return Z, Z, Z

    V = xp.asarray(viewpoints, dtype=xp.float64)
    vx = V[:, 0][:, None, None]
    vy = V[:, 1][:, None, None]
    Xg = X[None, :, :]
    Yg = Y[None, :, :]

    dx = _wrap_delta(Xg - vx, grid_size)
    dy = _wrap_delta(Yg - vy, grid_size)
    r2 = dx * dx + dy * dy + eps

    # NEW: ensure need_map is on the same backend (CuPy or NumPy)
    need = xp.asarray(need_map, dtype=xp.float64)

    Phi = (need[None, :, :] * 0.5 * xp.log(r2)).sum(axis=0)
    Fx  = (need[None, :, :] * (dx / r2)).sum(axis=0) * k_attr
    Fy  = (need[None, :, :] * (dy / r2)).sum(axis=0) * k_attr


    return Phi, Fx, Fy


def attraction_force_at_points(viewpoints, X, Y, need_map,
                               k_attr=10.0, grid_size=None):
    H, W = X.shape
    if viewpoints is None or len(viewpoints) == 0:
        return xp.zeros((0, 2), dtype=xp.float64)

    V = xp.asarray(viewpoints, dtype=xp.float64)
    vx = V[:, 0][:, None, None]
    vy = V[:, 1][:, None, None]
    Xg = X[None, :, :]
    Yg = Y[None, :, :]

    dx = _wrap_delta(Xg - vx, grid_size)
    dy = _wrap_delta(Yg - vy, grid_size)
    r2 = dx * dx + dy * dy + 1e-12

    # NEW: ensure need_map is on the same backend
    need = xp.asarray(need_map, dtype=xp.float64)
    w = need[None, :, :]

    fx = (w * (dx / r2)).sum(axis=(1, 2)) * k_attr
    fy = (w * (dy / r2)).sum(axis=(1, 2)) * k_attr
    return xp.stack([fx, fy], axis=1)



def attraction_force(grid_x, grid_y, center, k_attr=10.0, epsilon=1e-6):
    Phi, Fx, Fy = compute_log_potential(
        grid_x, grid_y, xp.asarray(center)[None, :],
        need_map=xp.ones_like(grid_x, dtype=xp.float64),
        k_attr=k_attr, eps=epsilon, grid_size=None
    )
    return Fx, Fy
