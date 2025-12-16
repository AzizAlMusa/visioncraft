# File: postprocessing/exp1_gain_ratio_sweep.py
"""
Experiment 1: stability vs attraction/repulsion gain ratio (k_attr / k_rep).

- Runs headless 2D torus simulation with N=8 viewpoints.
- Ratios gamma ∈ [1e-3, 1e3] (log-spaced, 100 points).
- For each gamma:
    * Run num_steps = 400.
    * On the last tail_len = 50 steps, compute Jacobian J(V)
      via numerical finite differences and extract:
        - lambda_max_real
        - spectral_radius(I + eta J)
    * Aggregate with medians (and some extra stats).
- Saves:
    logs/exp1_gain_ratio_summary.csv
    figures/exp1_lambda_max_vs_ratio.png
    figures/exp1_spectral_radius_vs_ratio.png

Run from the project root (where main.py lives):

    python -m postprocessing.exp1_gain_ratio_sweep

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Project imports (adjust if your package layout is a bit different)
from config.sim_config import SimConfig
from map.base_map import BaseMap
from entities.viewpoint import Viewpoints
from simulation.optimizer import Optimizer
from simulation.simulator import Simulator
from visibility.need_map import compute_need_map
from simulation.stability_logger import (
    numerical_jacobian,
    eig_metrics,
    torus_spacing_metrics,
)

# Optional GPU support detection
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
LOG_DIR = os.path.join(ROOT, "logs")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Helper: build a simulator + force_fn for a given config
# ---------------------------------------------------------------------

def make_sim_and_force_fn(cfg):
    """
    Build BaseMap, Viewpoints, Optimizer, Simulator and a CPU-friendly
    force_fn(positions_cpu) that matches the simulator's backend (NumPy/CuPy).
    """
    base_map = BaseMap(cfg.grid_size)

    # Force N=cfg.num_viewpoints viewpoints; overlap_test=False for speed
    viewpoints = Viewpoints(
        num=cfg.num_viewpoints,
        grid_size=cfg.grid_size,
        seed=cfg.seed,
        overlap_test=False,
    )

    opt = Optimizer(
        name=cfg.optimizer,
        step_size=cfg.step_size,
        beta1=cfg.adam_beta1,
        beta2=cfg.adam_beta2,
        eps=cfg.adam_eps,
        gain=cfg.adam_gain,
    )

    sim = Simulator(
        base_map,
        viewpoints,
        k_attr=cfg.k_attr,
        k_rep=cfg.k_rep,
        sigma_rep=cfg.sigma_rep,
        amp_rep=cfg.amp_rep,
        fov_radius=cfg.fov_radius,
        beta_vis=cfg.beta_vis,
        grid_size=cfg.grid_size,
        optimizer=opt,
    )

    X, Y = base_map.get_meshgrid()

    def force_fn(positions_cpu):
        """
        positions_cpu: (N,2) NumPy array in CPU memory.
        Returns forces as a NumPy array (for FD Jacobian).
        """
        # Recompute need_map for these hypothetical positions
        need_map_local = compute_need_map(
            X,
            Y,
            positions_cpu,
            cfg.fov_radius,
            beta=cfg.beta_vis,
            K_required=1.0,
            tau_clip=1.0,
            grid_size=cfg.grid_size,
        )

        # Match simulator backend (NumPy/CuPy)
        if HAS_CUPY and isinstance(sim.viewpoints.positions, cp.ndarray):
            positions_backend = cp.asarray(positions_cpu)
            need_map_backend = cp.asarray(need_map_local)
        else:
            positions_backend = positions_cpu
            need_map_backend = need_map_local

        forces_backend = sim.forces_given_positions(
            positions_backend, need_map_backend
        )

        # Always return NumPy for the FD Jacobian
        if HAS_CUPY and isinstance(forces_backend, cp.ndarray):
            return cp.asnumpy(forces_backend)
        return np.asarray(forces_backend)

    return base_map, viewpoints, sim, force_fn


# ---------------------------------------------------------------------
# Core runner: one gamma (k_attr / k_rep) -> summary metrics
# ---------------------------------------------------------------------

def run_single_ratio(
    gamma,
    num_steps=400,
    tail_len=50,
    jacobian_eps=1e-3,
):
    """
    Run a single simulation with gain ratio gamma = k_attr / k_rep.
    Returns a dict of aggregated stability metrics (medians over tail).
    """
    assert tail_len <= num_steps, "tail_len must be <= num_steps"

    cfg = SimConfig()

    # --- experiment-specific overrides ---
    cfg.num_steps = num_steps  # just for reference; we loop ourselves

    base_k_rep = cfg.k_rep
    cfg.k_rep = base_k_rep
    cfg.k_attr = gamma * base_k_rep

    print(
        f"[Exp1] gamma={gamma:.3e}  "
        f"k_attr={cfg.k_attr:.3f}  k_rep={cfg.k_rep:.3f}"
    )

    base_map, viewpoints, sim, force_fn = make_sim_and_force_fn(cfg)
    X, Y = base_map.get_meshgrid()

    # Storage for tail statistics
    lambda_tail = []
    rho_tail = []
    min_spacing_tail = []

    for step in range(num_steps):
        # --- compute fields + forces (does NOT move viewpoints) ---
        results = sim.step(i_active=0)

        # Need map for the *current* true positions
        need_map = compute_need_map(
            X,
            Y,
            viewpoints.positions,
            cfg.fov_radius,
            beta=cfg.beta_vis,
            K_required=1.0,
            tau_clip=1.0,
            grid_size=cfg.grid_size,
        )

        # --- tail region: do Jacobian + eigen probe ---
        if step >= num_steps - tail_len:
            # Jacobian of continuous-time F at current positions
            J = numerical_jacobian(
                force_fn,
                viewpoints.positions,
                eps=jacobian_eps,
            )
            eig_info = eig_metrics(J, cfg.step_size)

            lambda_tail.append(eig_info["lambda_max_real"])
            rho_tail.append(eig_info["spectral_radius_I_plus_etaJ"])

            spacing = torus_spacing_metrics(
                viewpoints.positions, float(cfg.grid_size)
            )
            min_spacing_tail.append(spacing["min_spacing"])

        # --- motion update (explicit Euler or Adam, matching main.py) ---
        forces = results["forces"]

        if HAS_CUPY:
            # Fix backend mismatch if positions are CuPy and forces are NumPy
            if isinstance(viewpoints.positions, cp.ndarray) and isinstance(
                forces, np.ndarray
            ):
                forces = cp.asarray(forces)

        if cfg.optimizer == "adam":
            viewpoints.positions = sim.optimizer.step(
                viewpoints.positions, forces
            ) % cfg.grid_size
        else:
            viewpoints.step(forces, cfg.step_size)

    # Convert to arrays for robust aggregation
    lambda_tail = np.asarray(lambda_tail, dtype=float)
    rho_tail = np.asarray(rho_tail, dtype=float)
    min_spacing_tail = np.asarray(min_spacing_tail, dtype=float)

    # Guard against any NaNs (e.g. if something blew up):
    lambda_tail = lambda_tail[np.isfinite(lambda_tail)]
    rho_tail = rho_tail[np.isfinite(rho_tail)]
    min_spacing_tail = min_spacing_tail[np.isfinite(min_spacing_tail)]

    # If everything went totally off the rails, fall back to NaNs
    if lambda_tail.size == 0:
        lambda_tail = np.array([np.nan])
    if rho_tail.size == 0:
        rho_tail = np.array([np.nan])
    if min_spacing_tail.size == 0:
        min_spacing_tail = np.array([np.nan])

    summary = {
        "gamma": gamma,
        "k_attr": cfg.k_attr,
        "k_rep": cfg.k_rep,
        # continuous-time eigen info
        "lambda_max_real_median": float(np.median(lambda_tail)),
        "lambda_max_real_mean": float(np.mean(lambda_tail)),
        "lambda_max_real_min": float(np.min(lambda_tail)),
        "lambda_max_real_max": float(np.max(lambda_tail)),
        "frac_cts_stable_tail": float((lambda_tail < 0.0).mean()),
        # discrete-time eigen info
        "spectral_radius_median": float(np.median(rho_tail)),
        "spectral_radius_mean": float(np.mean(rho_tail)),
        "spectral_radius_max": float(np.max(rho_tail)),
        "frac_disc_stable_tail": float((rho_tail < 1.0).mean()),
        # spacing (for context)
        "min_spacing_tail_mean": float(np.mean(min_spacing_tail)),
        "min_spacing_tail_min": float(np.min(min_spacing_tail)),
    }

    print(
        f"    -> median λ_max={summary['lambda_max_real_median']:.3f}  "
        f"median ρ={summary['spectral_radius_median']:.3f}  "
        f"cts_stable_tail={summary['frac_cts_stable_tail']:.2f}  "
        f"disc_stable_tail={summary['frac_disc_stable_tail']:.2f}"
    )

    return summary


# ---------------------------------------------------------------------
# Main sweep + plotting
# ---------------------------------------------------------------------

def main():
    # 100 log-spaced ratios from 1e-3 to 1e3
    gammas = np.logspace(-3, 3, 100)

    summaries = []
    for idx, gamma in enumerate(gammas):
        print(f"\n=== Experiment 1: run {idx+1}/{len(gammas)} / gamma={gamma:.6g} ===")
        summary = run_single_ratio(gamma)
        summaries.append(summary)

    df = pd.DataFrame(summaries)
    cfg = SimConfig()
    FILE_NAME = f"exp1_gain_ratio_sweep_N{cfg.num_viewpoints}.csv"
    summary_csv = os.path.join(LOG_DIR, FILE_NAME)
    df.to_csv(summary_csv, index=False)
    print(f"\n[Exp1] Summary saved to {summary_csv}")

    # ----------------------
    # Plot 1: λ_max vs ratio
    # ----------------------
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.semilogx(
        df["gamma"].values,
        df["lambda_max_real_median"].values,
        marker="o",
        linewidth=1.5,
    )
    ax1.axhline(0.0, linestyle="--", linewidth=1.0)
    ax1.set_xlabel(r"gain ratio $\gamma = k_{\mathrm{attr}} / k_{\mathrm{rep}}$")
    ax1.set_ylabel(r"median $\max \Re(\lambda)$ (tail)")
    ax1.set_title("Experiment 1: local continuous-time stability vs gain ratio")
    ax1.grid(True, which="both", linestyle=":", linewidth=0.5)

    fig1.tight_layout()
    fig1_path = os.path.join(FIG_DIR, "exp1_lambda_max_vs_ratio.png")
    fig1.savefig(fig1_path, dpi=200)
    print(f"[Exp1] Saved {fig1_path}")

    # ---------------------------------------
    # Plot 2: spectral radius vs ratio (disc)
    # ---------------------------------------
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.semilogx(
        df["gamma"].values,
        df["spectral_radius_median"].values,
        marker="o",
        linewidth=1.5,
    )
    ax2.axhline(1.0, linestyle="--", linewidth=1.0)
    ax2.set_xlabel(r"gain ratio $\gamma = k_{\mathrm{attr}} / k_{\mathrm{rep}}$")
    ax2.set_ylabel(r"median $\rho(I + \eta J)$ (tail)")
    ax2.set_title("Experiment 1: local discrete-time stability vs gain ratio")
    ax2.grid(True, which="both", linestyle=":", linewidth=0.5)

    fig2.tight_layout()
    fig2_path = os.path.join(FIG_DIR, "exp1_spectral_radius_vs_ratio.png")
    fig2.savefig(fig2_path, dpi=200)
    print(f"[Exp1] Saved {fig2_path}")


if __name__ == "__main__":
    main()
