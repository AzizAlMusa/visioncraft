# map/interest_map.py
"""
Interest Map
------------
Defines regions of higher importance on top of the base map,
such as circular or band-shaped areas, matching the logic
in overlap_anistropic (disk:, band:,...).

Author: Abdulaziz (overlap_sim)
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from map.base_map import create_base_map


def add_interest_disk(base, cx, cy, r, intensity=2.0, smooth_sigma=3.0):
    """
    Add a circular region of interest.
    """
    grid_size = base.shape[0]
    yy, xx = np.mgrid[0:grid_size, 0:grid_size]
    dist2 = (xx - cx)**2 + (yy - cy)**2
    mask = dist2 <= r**2
    new_map = base.copy()
    new_map[mask] = intensity
    if smooth_sigma > 0:
        new_map = gaussian_filter(new_map, smooth_sigma)
    return new_map


def add_interest_band(base, theta_deg=45.0, offset=0.0, width=15.0,
                      intensity=2.0, smooth_sigma=3.0):
    """
    Add a diagonal band of interest.
    """
    grid_size = base.shape[0]
    yy, xx = np.mgrid[0:grid_size, 0:grid_size]
    th = np.deg2rad(theta_deg)
    n = np.array([np.cos(th), np.sin(th)])
    proj = xx * n[0] + yy * n[1]
    mask = np.abs(proj - offset) <= (width / 2.0)
    new_map = base.copy()
    new_map[mask] = intensity
    if smooth_sigma > 0:
        new_map = gaussian_filter(new_map, smooth_sigma)
    return new_map


def normalize_map(M):
    """Normalize to [1, max]."""
    M = np.asarray(M, dtype=np.float64)
    minv, maxv = M.min(), M.max()
    if maxv - minv < 1e-12:
        return np.ones_like(M)
    return 1.0 + (M - minv) / (maxv - minv)


def build_interest_map(grid_size=100, mode="none", **kwargs):
    """
    Build an interest map overlay on top of the base map.
    """
    base = create_base_map(grid_size)

    if mode == "none":
        return base
    elif mode == "disk":
        M = add_interest_disk(base, **kwargs)
    elif mode == "band":
        M = add_interest_band(base, **kwargs)
    else:
        raise ValueError(f"Unknown interest mode: {mode}")

    return normalize_map(M)
