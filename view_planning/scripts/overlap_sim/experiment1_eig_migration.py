#!/usr/bin/env python3
"""
Experiment 1: Eigenvalue migration vs attraction/repulsion gain ratio.

For each gain ratio gamma = k_attr / k_rep:
  1) Run the torus simulation headlessly until approximate convergence.
  2) At the final configuration, build the full Jacobian J using
     finite differences on the true force function (including need-map).
  3) Compute all eigenvalues of J and write them to a CSV file.

Output:
  logs/exp1_eig_migration.csv

Columns:
  run_id, gamma, k_attr, k_rep, n,
  eig_index, lambda_real, lambda_imag,
  min_spacing, lambda_max_real
"""

import os
import csv
import numpy as np
import cupy as cp

from config.sim_config import SimConfig
from entities.viewpoint import Viewpoints        # :contentReference[oaicite:1]{index=1}
from map.base_map import BaseMap                 # :contentReference[oaicite:2]{index=2}
from simulation.simulator import Simulator       # :contentReference[oaicite:3]{index=3}
from visibility.need_map import compute_need_map

# ============================================================
# Experiment parameters (edit these)
# ============================================================
N_VIEWPOINTS = 8

# Gain ratios gamma = k_attr / k_rep to explore
# GAMMAS = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 1000.0]
GAMMAS = np.logspace(-3, 3, num=100).tolist()

MAX_STEPS = 200           # max iterations per run
VEL_THRESH = 1e-2         # mean force magnitude threshold for convergence
STAGNATION_STEPS = 20     # how many consecutive low-velocity steps to declare convergence
FD_EPS = 1e-3             # finite-difference step for Jacobian


# ============================================================
# Helpers
# ============================================================
def as_numpy(x):
    """Convert CuPy array to NumPy if needed."""
    if isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def min_torus_spacing(positions, grid_size):
    """Minimum wrapped distance between any pair of viewpoints on the torus."""
    P = as_numpy(positions)
    n = P.shape[0]
    if n < 2:
        return 0.0

    min_d = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            dx = P[i, 0] - P[j, 0]
            dy = P[i, 1] - P[j, 1]

            # wrap on torus
            dx = dx - np.round(dx / grid_size) * grid_size
            dy = dy - np.round(dy / grid_size) * grid_size

            d = np.sqrt(dx * dx + dy * dy)
            if d < min_d:
                min_d = d
    return float(min_d)


def make_force_fn(sim, base_map, cfg):
    """
    Build a CPU-side force function F(P) that:
      - takes a NumPy array positions_cpu (N,2),
      - recomputes need_map for those positions,
      - calls sim.forces_given_positions with correct backend,
      - returns NumPy forces (N,2).
    Mirrors the pattern used in main.py. :contentReference[oaicite:4]{index=4}
    """
    X, Y = base_map.get_meshgrid()

    def force_fn(positions_cpu: np.ndarray) -> np.ndarray:
        # Ensure NumPy array
        positions_cpu = np.asarray(positions_cpu, dtype=np.float64)

        # Need map for these positions
        need_map_local = compute_need_map(
            X, Y,
            positions_cpu,
            cfg.fov_radius,
            beta=cfg.beta_vis,
            K_required=1.0,
            tau_clip=1.0,
            grid_size=cfg.grid_size
        )

        # Match simulator backend (CuPy or NumPy)
        gpu_backend = isinstance(sim.viewpoints.positions, cp.ndarray)

        if gpu_backend:
            positions_backend = cp.asarray(positions_cpu)
            need_map_backend = cp.asarray(need_map_local)
        else:
            positions_backend = positions_cpu
            need_map_backend = need_map_local

        forces_backend = sim.forces_given_positions(
            positions_backend, need_map_backend
        )

        return as_numpy(forces_backend)

    return force_fn


def compute_jacobian_fd(force_fn, positions_cpu, eps=1e-3):
    """
    Finite-difference Jacobian of F at positions_cpu.

    positions_cpu: (N,2) NumPy array.
    Returns J of shape (2N, 2N) where F is flattened as (x1,y1,x2,y2,...).
    """
    P0 = np.asarray(positions_cpu, dtype=np.float64)
    N = P0.shape[0]
    dim = 2
    D = N * dim

    J = np.zeros((D, D), dtype=np.float64)

    # Baseline forces not strictly needed, but you can cache if desired
    # F0 = force_fn(P0).reshape(-1)

    for j in range(N):
        for d in range(dim):
            # Perturb coordinate j,d
            E = np.zeros_like(P0)
            E[j, d] = eps

            F_plus = force_fn(P0 + E).reshape(-1)
            F_minus = force_fn(P0 - E).reshape(-1)

            col = (F_plus - F_minus) / (2.0 * eps)
            col_index = j * dim + d
            J[:, col_index] = col

    return J


# ============================================================
# Main experiment
# ============================================================
def run_experiment():
    os.makedirs("logs", exist_ok=True)
    out_path = os.path.join("logs", "exp1_eig_migration.csv")

    # Prepare CSV writer
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id",
            "gamma",
            "k_attr",
            "k_rep",
            "n",
            "eig_index",
            "lambda_real",
            "lambda_imag",
            "min_spacing",
            "lambda_max_real"
        ])

        run_id = 0

        for gamma in GAMMAS:
            run_id += 1
            print(f"\n=== Experiment 1: run {run_id} / gamma={gamma} ===")

            # --------------------------------------------------
            # Build a fresh config & simulation for this gamma
            # --------------------------------------------------
            cfg = SimConfig()
            cfg.num_viewpoints = N_VIEWPOINTS

            # Choose k_rep as baseline, scale k_attr by gamma
            k_rep = cfg.k_rep
            k_attr = gamma * k_rep
            cfg.k_attr = k_attr

            print(f"[Config] n={cfg.num_viewpoints}  "
                  f"k_attr={cfg.k_attr:.4f}  k_rep={k_rep:.4f}  "
                  f"step_size={cfg.step_size:.4f}")

            base_map = BaseMap(cfg.grid_size)
            viewpoints = Viewpoints(
                num=cfg.num_viewpoints,
                grid_size=cfg.grid_size,
                seed=cfg.seed
            )

            sim = Simulator(
                base_map, viewpoints,
                k_attr=cfg.k_attr,
                k_rep=k_rep,
                sigma_rep=cfg.sigma_rep,
                amp_rep=cfg.amp_rep,
                fov_radius=cfg.fov_radius,
                beta_vis=cfg.beta_vis,
                grid_size=cfg.grid_size,
                optimizer=None
            )

            force_fn = make_force_fn(sim, base_map, cfg)

            # --------------------------------------------------
            # Run until convergence (or MAX_STEPS)
            # --------------------------------------------------
            stagnation_count = 0
            for step in range(MAX_STEPS):
                results = sim.step(i_active=None)
                forces = results["forces"]

                # Update positions (vanilla gradient ascent)
                viewpoints.step(forces, step_size=cfg.step_size)

                # Check convergence based on mean force magnitude
                forces_np = as_numpy(forces)
                mean_force = np.linalg.norm(forces_np, axis=1).mean()

                if mean_force < VEL_THRESH:
                    stagnation_count += 1
                else:
                    stagnation_count = 0

                if stagnation_count >= STAGNATION_STEPS:
                    print(f"[Converged] step={step}  "
                          f"mean_force={mean_force:.3e}")
                    break

                if (step + 1) % 50 == 0:
                    print(f"[Progress] step={step+1}  "
                          f"mean_force={mean_force:.3e}")

            # Final state
            P_final = as_numpy(viewpoints.positions)
            min_spacing = min_torus_spacing(P_final, cfg.grid_size)

            # --------------------------------------------------
            # Jacobian and eigenvalues at final configuration
            # --------------------------------------------------
            print("[Jacobian] computing finite-difference Jacobian...")
            J = compute_jacobian_fd(force_fn, P_final, eps=FD_EPS)
            eigvals = np.linalg.eigvals(J)

            lambda_max_real = float(np.max(eigvals.real))
            print(f"[Jacobian] max Re(lambda) = {lambda_max_real:.4f}")

            # --------------------------------------------------
            # Write one row per eigenvalue
            # --------------------------------------------------
            for idx, lam in enumerate(eigvals):
                writer.writerow([
                    run_id,
                    gamma,
                    k_attr,
                    k_rep,
                    cfg.num_viewpoints,
                    idx,
                    float(lam.real),
                    float(lam.imag),
                    min_spacing,
                    lambda_max_real
                ])

    print(f"\n[Done] Saved Experiment 1 eigenvalues to {out_path}")


if __name__ == "__main__":
    run_experiment()
