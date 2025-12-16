# physics/need_map.py
try:
    import cupy as np
    GPU = True
except ImportError:
    import numpy as np
    GPU = False


def _wrap_delta(delta, grid_size):
    """Toroidal minimal-image convention for pairwise deltas."""
    if grid_size is None:
        return delta
    half = grid_size / 2.0
    return np.where(np.abs(delta) > half,
                    delta - np.sign(delta) * grid_size,
                    delta)

def compute_need_map(X, Y, viewpoints, fov_radius, beta=20.0, K_required=1.0,
                     tau_clip=1.0, grid_size=None):
    """
    Vanilla coverage deficit:
        vis = sum_k sigmoid( beta * (d_k / fov - 1) )
        deficit = max(0, 1 - vis / K_required)
        (optionally hard-clip when sufficiently visible)

    Returns
    -------
    need : float32 array, same shape as X/Y, in [0,1]
    """
    H, W = X.shape
    if viewpoints is None or len(viewpoints) == 0:
        return np.ones((H, W), dtype=np.float32)

    # (N, 1, 1) viewpoint coords broadcast over (H, W)
    V = np.asarray(viewpoints, dtype=np.float64)
    vx = V[:, 0][:, None, None]
    vy = V[:, 1][:, None, None]

    # (1, H, W) grid coords
    Xg = X[None, :, :]
    Yg = Y[None, :, :]

    # Toroidal deltas, shape (N, H, W)
    dx = _wrap_delta(Xg - vx, grid_size)
    dy = _wrap_delta(Yg - vy, grid_size)

    # Distances and sigmoid visibility
    d = np.sqrt(dx * dx + dy * dy) + 1e-12
    s = 1.0 / (1.0 + np.exp(beta * (d / float(fov_radius) - 1.0)))  # (N, H, W)

    vis = s.sum(axis=0)  # (H, W)
    deficit = np.maximum(0.0, 1.0 - vis / float(K_required))

    # optional hard-clip: if already tau_clip*K_required visible, zero the need
    deficit = np.where(vis >= tau_clip * K_required, 0.0, deficit)

    return deficit.astype(np.float32)
