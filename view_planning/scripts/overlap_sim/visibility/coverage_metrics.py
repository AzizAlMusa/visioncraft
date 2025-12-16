"""
Coverage Metrics (GPU/CPU)
--------------------------
Provides two complementary coverage modes:

1. Grid-based coverage  (using full visibility grid)
2. Probe-based coverage (sampled points across the space)

All metrics are consistent with the sigmoid visibility model:
    s_i(p) = 1 / (1 + exp(beta * (d_i / fov_radius - 1)))

Metrics returned:
- per_viewpoint:  mean visibility per viewpoint
- total_coverage: union coverage across all viewpoints
- novel:          unique contribution per viewpoint
- overlap_matrix: NxN pairwise shared visibility
- coverage_map:   dense grid or sampled probes

Automatically uses CuPy if available.
"""

try:
    import cupy as np
    GPU = True
except ImportError:
    import numpy as np
    GPU = False


# ================================================================
# Optional: probe generator
# ================================================================
def make_probes(grid_size, n_probes=5000, method="grid", seed=0, jitter=True):
    """
    Generate probe points within [0, grid_size)^2 for coverage estimation.
    """
    import numpy as _np
    rng = _np.random.default_rng(seed)

    if method == "grid":
        m = int(_np.sqrt(n_probes))
        m = max(1, m)
        xs = _np.linspace(0.5, grid_size - 0.5, m)
        ys = _np.linspace(0.5, grid_size - 0.5, m)
        Xg, Yg = _np.meshgrid(xs, ys)
        P = _np.stack([Xg.ravel(), Yg.ravel()], axis=1)
        if jitter:
            cell = grid_size / m
            P += rng.uniform(-0.3 * cell, 0.3 * cell, size=P.shape)
        if len(P) > n_probes:
            P = P[:n_probes]
        return P.astype(_np.float64)

    elif method == "random":
        P = rng.uniform(0.0, grid_size, size=(n_probes, 2))
        return P.astype(_np.float64)
    else:
        raise ValueError("method must be 'grid' or 'random'")


# ================================================================
# Core Coverage Function (Grid or Probe)
# ================================================================
def compute_coverage_metrics(
    X,
    Y,
    viewpoints,
    fov_radius,
    beta=20.0,
    grid_size=None,
    probes=None,
    hard_threshold=False,
    thresh=0.5,
):
    """
    Compute continuous or probe-based coverage metrics.

    Parameters
    ----------
    X, Y : np.ndarray or cp.ndarray
        Domain grid (ignored if probes provided)
    viewpoints : (N,2) array
    fov_radius : float
    beta : float
    grid_size : float
    probes : (P,2) optional array of probe coordinates
        If provided, coverage is computed on these points.
    hard_threshold : bool
        If True, coverage is binary (s > thresh)
    thresh : float
        Visibility cutoff for binary mode

    Returns
    -------
    dict with:
        coverage_map, per_viewpoint, novel, total_coverage, overlap_matrix
        (and 'probes' if probe-based)
    """
    if probes is not None:
        # -----------------------------------------------------------
        # --- Probe-based coverage ---------------------------------
        # -----------------------------------------------------------
        P = np.asarray(probes, dtype=np.float64)
        V = np.asarray(viewpoints, dtype=np.float64)
        N = V.shape[0]
        Pn = P.shape[0]

        if N == 0 or Pn == 0:
            zeros = np.zeros((Pn,), dtype=np.float32)
            return dict(
                coverage_map=zeros,
                per_viewpoint=[],
                novel=[],
                total_coverage=0.0,
                overlap_matrix=np.zeros((0, 0), dtype=np.float32),
                probes=P,
            )

        diff = P[None, :, :] - V[:, None, :]  # (N,P,2)
        if grid_size is not None:
            half = grid_size / 2.0
            diff = np.where(np.abs(diff) > half, diff - np.sign(diff) * grid_size, diff)

        d = np.sqrt(np.sum(diff**2, axis=2)) + 1e-12
        s = 1.0 / (1.0 + np.exp(beta * (d / fov_radius - 1.0)))  # (N,P)

        if hard_threshold:
            s = (s > thresh).astype(np.float32)

        # --- total (union) coverage ---
        coverage_total = 1.0 - np.prod(1.0 - s, axis=0)
        total_coverage = float(np.mean(coverage_total))
        per_viewpoint = [float(np.mean(s[i])) for i in range(N)]

        # --- overlap matrix ---
        overlap_matrix = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                overlap_matrix[i, j] = float(np.mean(s[i] * s[j]))

        # --- novel ---
        novel = []
        for i in range(N):
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            others = np.max(s[mask], axis=0) if N > 1 else np.zeros_like(s[i])
            unique_mask = s[i] * (1 - others)
            novel.append(float(np.mean(unique_mask)))

        if GPU:
            import cupy as cp
            coverage_total = cp.asnumpy(coverage_total)
            overlap_matrix = cp.asnumpy(overlap_matrix)

        return dict(
            coverage_map=coverage_total.astype(np.float32),
            per_viewpoint=per_viewpoint,
            novel=novel,
            total_coverage=total_coverage,
            overlap_matrix=overlap_matrix,
            probes=P,
        )

    else:
        # -----------------------------------------------------------
        # --- Grid-based coverage ----------------------------------
        # -----------------------------------------------------------
        H, W = X.shape
        N = len(viewpoints)
        if N == 0:
            zeros = np.zeros((H, W), dtype=np.float32)
            return dict(
                coverage_map=zeros,
                per_viewpoint=[],
                novel=[],
                total_coverage=0.0,
                overlap_matrix=np.zeros((0, 0)),
            )

        X = np.asarray(X)
        Y = np.asarray(Y)
        V = np.asarray(viewpoints, dtype=np.float64)
        half = grid_size / 2.0 if grid_size else None

        vx = V[:, 0][:, None, None]
        vy = V[:, 1][:, None, None]
        dx = X[None, :, :] - vx
        dy = Y[None, :, :] - vy

        if grid_size:
            dx = np.where(np.abs(dx) > half, dx - np.sign(dx) * grid_size, dx)
            dy = np.where(np.abs(dy) > half, dy - np.sign(dy) * grid_size, dy)

        d = np.sqrt(dx**2 + dy**2) + 1e-12
        vis_stack = 1.0 / (1.0 + np.exp(beta * (d / fov_radius - 1.0)))  # (N,H,W)

        if hard_threshold:
            vis_stack = (vis_stack > thresh).astype(np.float32)

        coverage_total = 1.0 - np.prod(1.0 - vis_stack, axis=0)
        total_coverage = float(np.mean(coverage_total))
        per_viewpoint = [float(np.mean(vis_stack[i])) for i in range(N)]

        overlap_matrix = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                overlap_matrix[i, j] = float(np.mean(vis_stack[i] * vis_stack[j]))

        novel = []
        for i in range(N):
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            others_max = np.max(vis_stack[mask], axis=0) if N > 1 else np.zeros_like(vis_stack[i])
            uniq = vis_stack[i] * (1 - others_max)
            novel.append(float(np.mean(uniq)))

        if GPU:
            import cupy as cp
            coverage_total = cp.asnumpy(coverage_total)
            overlap_matrix = cp.asnumpy(overlap_matrix)

        return dict(
            coverage_map=coverage_total.astype(np.float32),
            per_viewpoint=per_viewpoint,
            novel=novel,
            total_coverage=total_coverage,
            overlap_matrix=overlap_matrix,
        )
