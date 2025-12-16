#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU profiling script
--------------------
Quick benchmark for compute_need_map + compute_log_potential
to verify actual GPU speed (excluding visualization).
"""

import time
import cupy as cp

# Correct imports
from visibility.need_map import compute_need_map
from physics.attraction import compute_log_potential

# --- Parameters ---
grid_size = 400
num_vps = 10
fov_radius = 20.0
beta_vis = 20.0
k_attr = 10.0

# --- Setup ---
X, Y = cp.meshgrid(cp.arange(grid_size), cp.arange(grid_size))
V = cp.random.rand(num_vps, 2) * grid_size
need = cp.ones((grid_size, grid_size), dtype=cp.float32)

# --- Warm-up ---
_ = compute_need_map(X, Y, V, fov_radius, beta_vis, grid_size=grid_size)
_ = compute_log_potential(X, Y, V, need, k_attr=k_attr, grid_size=grid_size)
cp.cuda.Stream.null.synchronize()

# --- Benchmark ---
t0 = time.time()
for _ in range(50):
    need_map = compute_need_map(X, Y, V, fov_radius, beta_vis, grid_size=grid_size)
    phi, Fx, Fy = compute_log_potential(X, Y, V, need_map, k_attr=k_attr, grid_size=grid_size)
cp.cuda.Stream.null.synchronize()
t1 = time.time()

print(f"Grid: {grid_size}x{grid_size}, Viewpoints: {num_vps}")
print(f"Average time per iteration: {(t1 - t0)/50:.4f} sec ({50/(t1 - t0):.1f} iter/sec)")
