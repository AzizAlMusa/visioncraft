# simulation/diagnostics.py
"""
Step-by-step stability diagnostics.

For a single step we compute:
- J_full : Jacobian of the *full* vector field (need map recomputed inside force_fn)
- J_geo  : Jacobian with the need map frozen (only attraction+repulsion geometry)
- J_need : J_full - J_geo -> contribution from need-map changes
- J_attr : geometric Jacobian coming only from the attraction forces
- J_rep  : geometric Jacobian coming only from the repulsion forces

We also look at how the need map changes if we nudge the "worst" coordinate
(the column of J_need with largest norm) by ±eps.
"""

import numpy as np

try:
    import cupy as cp
    GPU = True
except ImportError:
    cp = None
    GPU = False

from visibility.need_map import compute_need_map


def _to_numpy(x):
    """CuPy → NumPy helper (no-op for NumPy)."""
    if GPU and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x, dtype=np.float64)


def _central_fd_jacobian(force_fn, positions, eps):
    """
    Central finite-difference Jacobian for a force function:

        force_fn: R^{N×2} -> R^{N×2}

    returns a (2N × 2N) matrix in NumPy.
    """
    P0 = _to_numpy(positions)
    N = P0.shape[0]
    dim = 2 * N

    J = np.zeros((dim, dim), dtype=np.float64)

    for k in range(dim):
        dP = np.zeros_like(P0)
        i = k // 2     # viewpoint index
        j = k % 2      # 0: x, 1: y
        dP[i, j] = eps

        P_plus = P0 + dP
        P_minus = P0 - dP

        F_plus = _to_numpy(force_fn(P_plus)).reshape(-1)
        F_minus = _to_numpy(force_fn(P_minus)).reshape(-1)

        J[:, k] = (F_plus - F_minus) / (2.0 * eps)

    return J


def _forces_split(sim, P, need_np):
    """
    Helper: for given positions P and need map (NumPy),
    return (F_attr, F_rep) as NumPy arrays.

    This temporarily overwrites sim.viewpoints.positions and restores it.
    """
    P_np = _to_numpy(P)

    # we need to go through the simulator helper that returns attraction+repulsion
    backup = sim.viewpoints.positions
    try:
        sim.viewpoints.positions = P_np
        res = sim._compute_forces_with_need(need_np)
    finally:
        sim.viewpoints.positions = backup

    F_attr = _to_numpy(res["forces_attr"])
    F_rep = _to_numpy(res["forces_rep"])
    return F_attr, F_rep


def debug_step(step, sim, cfg, X, Y, positions, need_map, eps, force_fn):
    """
    Run a heavy but very explicit diagnostic for ONE time step.

    Parameters
    ----------
    step      : int
    sim       : Simulator instance
    cfg       : config (used only for k_rep, fov_radius, beta_vis, grid_size)
    X, Y      : meshgrid (same as sim.X, sim.Y)
    positions : current viewpoints positions (CuPy or NumPy)
    need_map  : current need map for these positions
    eps       : finite-difference step (same as jacobian_eps)
    force_fn  : the 'live' force function used in log_step_metrics
                (recomputes need_map inside)

    Returns a dict with all matrices and eigenvalues, and prints a summary.
    """
    P0 = _to_numpy(positions)
    need_np = _to_numpy(need_map)

    # ------------------------------------------------------------------
    # (1) Full Jacobian: this matches what your stability logger uses.
    # ------------------------------------------------------------------
    def F_full(P):
        return _to_numpy(force_fn(P))

    J_full = _central_fd_jacobian(F_full, P0, eps)
    eig_full = np.linalg.eigvals(J_full)

    # ------------------------------------------------------------------
    # (2) Geometric Jacobian with frozen need map (no coverage changes)
    # ------------------------------------------------------------------
    def F_geo(P):
        # attraction + k_rep * repulsion with fixed need_np
        backup = sim.viewpoints.positions
        try:
            sim.viewpoints.positions = _to_numpy(P)
            res = sim._compute_forces_with_need(need_np)
            return _to_numpy(res["forces"])
        finally:
            sim.viewpoints.positions = backup

    J_geo = _central_fd_jacobian(F_geo, P0, eps)
    eig_geo = np.linalg.eigvals(J_geo)

    # ------------------------------------------------------------------
    # (3) Need-map contribution: J_need = J_full - J_geo
    # ------------------------------------------------------------------
    J_need = J_full - J_geo
    eig_need = np.linalg.eigvals(J_need)

    # ------------------------------------------------------------------
    # (4) Split geometric Jacobian into attraction vs repulsion
    # ------------------------------------------------------------------
    def F_attr(P):
        F_attr_np, _ = _forces_split(sim, P, need_np)
        return F_attr_np

    def F_rep(P):
        # multiply by k_rep so that J_attr + J_rep ≈ J_geo
        _, F_rep_np = _forces_split(sim, P, need_np)
        return cfg.k_rep * F_rep_np

    J_attr = _central_fd_jacobian(F_attr, P0, eps)
    J_rep = _central_fd_jacobian(F_rep, P0, eps)
    eig_attr = np.linalg.eigvals(J_attr)
    eig_rep = np.linalg.eigvals(J_rep)

    def max_real(vals):
        return float(np.max(np.real(vals))) if vals.size else np.nan

    max_full = max_real(eig_full)
    max_geo = max_real(eig_geo)
    max_need = max_real(eig_need)
    max_attr = max_real(eig_attr)
    max_rep = max_real(eig_rep)

    # ------------------------------------------------------------------
    # (5) Find which *coordinate* in J_need is causing the biggest impact
    # ------------------------------------------------------------------
    col_norms = np.linalg.norm(J_need, axis=0)
    bad_col = int(np.argmax(col_norms))
    bad_vp = bad_col // 2
    bad_axis = "x" if bad_col % 2 == 0 else "y"

    # ------------------------------------------------------------------
    # (6) Measure how the need map changes when we nudge that coordinate
    # ------------------------------------------------------------------
    dP = np.zeros_like(P0)
    dP[bad_vp, bad_col % 2] = eps

    P_plus = P0 + dP
    P_minus = P0 - dP

    need_plus = compute_need_map(
        X, Y, P_plus,
        fov_radius=sim.fov_radius,
        beta=sim.beta_vis,
        grid_size=sim.grid_size,
    )
    need_minus = compute_need_map(
        X, Y, P_minus,
        fov_radius=sim.fov_radius,
        beta=sim.beta_vis,
        grid_size=sim.grid_size,
    )

    dn_plus = _to_numpy(need_plus) - need_np
    dn_minus = _to_numpy(need_minus) - need_np

    thr = 0.4  # "big" change in need
    frac_plus = float(np.mean(np.abs(dn_plus) > thr))
    frac_minus = float(np.mean(np.abs(dn_minus) > thr))
    max_jump = float(
        max(np.max(np.abs(dn_plus)), np.max(np.abs(dn_minus)))
    )

    # ------------------------------------------------------------------
    # (7) Human-readable summary
    # ------------------------------------------------------------------
    print("\n========== DEBUG STEP %04d ==========" % step)
    print(f"maxReλ full      = {max_full: .3e}")
    print(f"maxReλ frozen    = {max_geo: .3e}")
    print(f"maxReλ need only = {max_need: .3e}")
    print(f"maxReλ attraction= {max_attr: .3e}")
    print(f"maxReλ repulsion = {max_rep: .3e}")
    print(f"worst J_need column = {bad_col} "
          f"(viewpoint {bad_vp}, axis {bad_axis})")
    print(f"need jumps around that nudge: "
          f"max |Δn|={max_jump:.3f}, "
          f"frac(|Δn_plus|>0.4)={frac_plus:.3f}, "
          f"frac(|Δn_minus|>0.4)={frac_minus:.3f}")

    return {
        "J_full": J_full,
        "J_geo": J_geo,
        "J_need": J_need,
        "J_attr": J_attr,
        "J_rep": J_rep,
        "eig_full": eig_full,
        "eig_geo": eig_geo,
        "eig_need": eig_need,
        "eig_attr": eig_attr,
        "eig_rep": eig_rep,
        "maxRe_full": max_full,
        "maxRe_geo": max_geo,
        "maxRe_need": max_need,
        "maxRe_attr": max_attr,
        "maxRe_rep": max_rep,
        "bad_col": bad_col,
        "bad_viewpoint": bad_vp,
        "bad_axis": bad_axis,
        "need_jump_max": max_jump,
        "need_jump_frac_plus": frac_plus,
        "need_jump_frac_minus": frac_minus,
    }
