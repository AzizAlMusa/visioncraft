"""
Unified field computation (GPU-safe)
-----------------------------------
All arrays remain on CuPy if available.
"""

try:
    import cupy as xp
    GPU = True
except ImportError:
    import numpy as xp
    GPU = False

from physics.attraction import compute_log_potential
from physics.repulsion import compute_repulsion_field


def compute_fields(X, Y, viewpoints, need_map,
                   k_attr=10.0, k_rep=0.25, sigma_rep=10.0, amp_rep=100.0,
                   grid_size=None):
    potential, Fx_attr, Fy_attr = compute_log_potential(
        X, Y, viewpoints, need_map, k_attr=k_attr, grid_size=grid_size
    )
    Fx_rep, Fy_rep = compute_repulsion_field(
        X, Y, viewpoints, sigma=sigma_rep, amp=amp_rep, grid_size=grid_size
    )
    Fx_total = Fx_attr + k_rep * Fx_rep
    Fy_total = Fy_attr + k_rep * Fy_rep
    return potential, Fx_attr, Fy_attr, Fx_rep, Fy_rep, Fx_total, Fy_total


def compute_fields_from_viewpoint(X, Y, viewpoints, i_active, need_map,
                                  k_attr=10.0, k_rep=0.25,
                                  sigma_rep=10.0, amp_rep=100.0,
                                  grid_size=None):
    if viewpoints is None or len(viewpoints) == 0:
        Z = xp.zeros_like(X, dtype=xp.float64)
        return Z, Z, Z, Z, Z, Z, Z

    potential, _, _ = compute_log_potential(
        X, Y, viewpoints, need_map, k_attr=k_attr, grid_size=grid_size
    )

    v_active = viewpoints[i_active:i_active + 1]
    if i_active == 0:
        others = viewpoints[1:]
    elif i_active == len(viewpoints) - 1:
        others = viewpoints[:-1]
    else:
        others = xp.concatenate((viewpoints[:i_active], viewpoints[i_active + 1:]), axis=0)

    _, Fx_attr, Fy_attr = compute_log_potential(
        X, Y, v_active, need_map, k_attr=k_attr, grid_size=grid_size
    )
    Fx_rep, Fy_rep = compute_repulsion_field(
        X, Y, others, sigma=sigma_rep, amp=amp_rep, grid_size=grid_size
    )

    Fx_total = Fx_attr + k_rep * Fx_rep
    Fy_total = Fy_attr + k_rep * Fy_rep
    return potential, Fx_attr, Fy_attr, Fx_rep, Fy_rep, Fx_total, Fy_total
