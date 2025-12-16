#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main simulation script for overlap_sim with timing breakdown
"""
import sys
import select
import termios
import tty

import numpy as np
import cupy as cp

import time

from config.sim_config import SimConfig
from entities.viewpoint import Viewpoints
from map.base_map import BaseMap

from physics.fields import compute_fields_from_viewpoint

from simulation.simulator import Simulator
from simulation.optimizer import Optimizer
from simulation.insertion_manager import InsertionManager

from visualization.viewer import Viewer

from visualization.panels.potential_panel import PotentialPanel
from visualization.panels.vector_field_panel import VectorFieldPanel


from visibility.need_map import compute_need_map


from visibility.coverage_metrics import compute_coverage_metrics, make_probes

from simulation.stability_experiments import log_step_metrics
from simulation.stability_logger import log_step_metrics

from simulation.diagnostics import debug_step

# ============================================================
# Non-blocking keyboard listener (terminal)
# ============================================================
class KeyListener:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)

    def __enter__(self):
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def get_key(self):
        """Return pressed key if any, else None.

        - Regular keys: returns a 1-character string, e.g. 'q', 'i'
        - Arrow keys: returns 'UP', 'DOWN', 'LEFT', 'RIGHT'
        """
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if not dr:
            return None

        ch1 = sys.stdin.read(1)
        if ch1 != "\x1b":
            # normal single-key press
            return ch1

        # Possible escape sequence (arrow keys)
        # Try to read the next two bytes without blocking
        if not select.select([sys.stdin], [], [], 0)[0]:
            return None
        ch2 = sys.stdin.read(1)
        if not select.select([sys.stdin], [], [], 0)[0]:
            return None
        ch3 = sys.stdin.read(1)

        seq = ch1 + ch2 + ch3
        if seq == "\x1b[A":
            return "UP"
        if seq == "\x1b[B":
            return "DOWN"
        if seq == "\x1b[C":
            return "RIGHT"
        if seq == "\x1b[D":
            return "LEFT"

        # Unknown escape sequence – ignore
        return None





# ============================================================
# Configuration and Initialization
# ============================================================
cfg = SimConfig()
np.random.seed(cfg.seed + 1)


# ---------------------------------------------------------
# Manual experiment label: edit this string between runs
# ---------------------------------------------------------
alpha = round(cfg.k_attr / cfg.k_rep, 1)
EXPERIMENT_NAME = f"C_{alpha}"  # e.g. "A_N8", "B_eta_small", ...
# CSV_PATH = f"./logs/stability_{EXPERIMENT_NAME}.csv"
CSV_PATH = "./logs/stability_general_metrics.csv"
# Steps at which we want the heavy-duty diagnostics
DEBUG_STEPS = {206, 253, 280, 331, 374}  # adjust these to whatever steps looked weird


base_map = BaseMap(cfg.grid_size)
# viewpoints = Viewpoints(cfg.num_viewpoints, cfg.grid_size, seed=cfg.seed)
# viewpoints = Viewpoints(num=cfg.num_viewpoints, grid_size=cfg.grid_size, seed=cfg.seed, overlap_test=True)
viewpoints = Viewpoints(
    num=cfg.num_viewpoints,
    grid_size=cfg.grid_size,
    seed=cfg.seed,
    overlap_test=False
)
# viewer = Viewer(
#     [
#         PotentialPanel(grid_size=cfg.grid_size, fov_radius=cfg.fov_radius),
#         VectorFieldPanel(grid_size=cfg.grid_size),
#     ],
#     target_fps=cfg.target_fps,
#     record=cfg.record_video,
#     out_path=cfg.video_path
# )
viewer = Viewer(
    [
        PotentialPanel(grid_size=cfg.grid_size, fov_radius=cfg.fov_radius),
        VectorFieldPanel(grid_size=cfg.grid_size),
    ],
    title="OverlapSim – 1x2 Layout",
    size=(1200, 600),
    record=cfg.record_video,
    out_path=cfg.video_path
)


# --- ensure synchronization with the first viewpoint (fixes repulsion issue) ---
viewer.active_index = 0


def perturb_active_viewpoint(direction, viewpoints, viewer, delta=1.0, grid_size=100):
    """Nudge the currently active viewpoint by +/- delta in x/y."""
    idx = getattr(viewer, "active_index", None)
    if idx is None:
        return

    pos = viewpoints.positions
    if idx < 0 or idx >= pos.shape[0]:
        return

    # Make sure we're in a numpy / cupy array, but operate in-place
    dx, dy = 0.0, 0.0
    if direction == "UP":
        dy = -delta
    elif direction == "DOWN":
        dy = +delta
    elif direction == "LEFT":
        dx = -delta
    elif direction == "RIGHT":
        dx = +delta
    else:
        return

    pos[idx, 0] = (pos[idx, 0] + dx) % grid_size
    pos[idx, 1] = (pos[idx, 1] + dy) % grid_size



opt = Optimizer(
    name=cfg.optimizer,
    step_size=cfg.step_size,
    beta1=cfg.adam_beta1,
    beta2=cfg.adam_beta2,
    eps=cfg.adam_eps,
    gain=cfg.adam_gain,
)

insertion_mgr = InsertionManager(
    mode=cfg.insertion_mode,
    insert_interval=cfg.insert_interval,
    velocity_thresh=cfg.velocity_thresh,
    stagnation_steps=cfg.stagnation_steps
)

sim = Simulator(
    base_map, viewpoints,
    k_attr=cfg.k_attr, k_rep=cfg.k_rep,
    sigma_rep=cfg.sigma_rep, amp_rep=cfg.amp_rep,
    fov_radius=cfg.fov_radius, beta_vis=cfg.beta_vis,
    grid_size=cfg.grid_size,
    optimizer=opt
)

# analyzer = Analyzer(
#     grid_size=cfg.grid_size,
#     k_attr=cfg.k_attr,
#     k_rep=cfg.k_rep,
#     sigma=cfg.sigma_rep,
#     amp=cfg.amp_rep,
#     eta=cfg.step_size,             # for discrete-time eigen test
#     print_every=10,                # live line every 10 steps
#     jacobian_every=50,             # eigen probe every 50 steps (optional)
#     jacobian_fd_eps=1e-3,
#     jacobian_max_N=8               # keep small for speed
# )



sim.insertion_manager = insertion_mgr

# Initialize probe set once
probes = make_probes(cfg.grid_size, n_probes=8000, method="grid", seed=0)



print(f"[Init] grid={cfg.grid_size}  viewpoints={cfg.num_viewpoints}")
print(f"[Mode] insertion={cfg.insertion_mode}, record={cfg.record_video}")


# ============================================================
# Force callback for Jacobian probe (used by Analyzer)
# ============================================================
def force_fn(positions_cpu):
    """
    Compute total forces for arbitrary viewpoint positions.
    Ensures backend consistency with Simulator (NumPy or CuPy).
    """
    X, Y = base_map.get_meshgrid()

    need_map_local = compute_need_map(
        X, Y, positions_cpu, cfg.fov_radius,
        beta=cfg.beta_vis, K_required=1.0, tau_clip=1.0,
        grid_size=cfg.grid_size
    )

    # Match simulator backend
    try:
        import cupy as cp
        gpu_backend = isinstance(sim.viewpoints.positions, cp.ndarray)
    except ImportError:
        gpu_backend = False

    if gpu_backend:
        positions_backend = cp.asarray(positions_cpu)
        need_map_backend = cp.asarray(need_map_local)
    else:
        positions_backend = positions_cpu
        need_map_backend = need_map_local

    return sim.forces_given_positions(positions_backend, need_map_backend)



# ============================================================
# Main Simulation Loop (timed)
# ============================================================
try:
    with KeyListener() as key_listener:
        for step in range(cfg.num_steps):
            # Allow graceful quit via viewer
            if getattr(viewer, "quit", False):
                print("[Main] Quit requested — stopping simulation.")
                break

            t0 = time.time()

            # --- compute fields ---
            t1 = time.time()
            results = sim.step(i_active=viewer.active_index or 0)
            t2 = time.time()

            # --- per-viewpoint perspective vector fields (keep global potential from sim) ---
            X, Y = base_map.get_meshgrid()
            need_map = compute_need_map(X, Y, viewpoints.positions, cfg.fov_radius,
                                        beta=cfg.beta_vis, K_required=1.0, tau_clip=1.0,
                                        grid_size=cfg.grid_size)
            
            # coverage computation and statistics
            # coverage = compute_coverage_metrics(
            #     X, Y, viewpoints.positions, cfg.fov_radius,
            #     beta=cfg.beta_vis, grid_size=cfg.grid_size, probes=probes
            # )

            # print(f"[Coverage] total={coverage['total_coverage']:.3f} "
            #     f"per={ [round(x,3) for x in coverage['per_viewpoint']] } "
            #     f"novel={ [round(x,3) for x in coverage['novel']] } "
            #     f"overlap_mean={coverage['overlap_matrix'].mean():.3f}")

            # fields from the active viewpoint's perspective (attraction: active only, repulsion: others)
            _, Fx_attr_vp, Fy_attr_vp, Fx_rep_vp, Fy_rep_vp, Fx_total_vp, Fy_total_vp = \
                compute_fields_from_viewpoint(
                    X, Y,
                    viewpoints.positions,
                    viewer.active_index,
                    need_map,
                    k_attr=cfg.k_attr, k_rep=cfg.k_rep,
                    sigma_rep=cfg.sigma_rep, amp_rep=cfg.amp_rep,
                    grid_size=cfg.grid_size
                )

            # ============================================================
            # Terminal key injection: press 'i' in terminal to insert viewpoint
            # ============================================================
            key = key_listener.get_key()
            if key == "i":
                viewer.insertion_request = True
                print("[Terminal] 'i' pressed — manual viewpoint insertion requested.")

            # --- viewpoint insertion (manual) ---
            insertion_happened = False
            if viewer.insertion_request:
                new_pos = sim.insertion_manager.select_best_insertion(
                    results["potential"], cfg.grid_size
                )
                viewpoints.add_viewpoint(new_pos)
                viewer.insertion_request = False
                insertion_happened = True
                print(f"[Main] Added new viewpoint manually at {new_pos} (highest potential)")

            # --- motion update ---
            if not insertion_happened:
                forces = results["forces"]

                # --- ensure backend consistency (fixes CuPy/NumPy mismatch) ---
                if isinstance(viewpoints.positions, cp.ndarray) and isinstance(forces, np.ndarray):
                    forces = cp.asarray(forces)

                if cfg.optimizer == "adam":
                    viewpoints.positions = sim.optimizer.step(
                        viewpoints.positions, forces
                    ) % cfg.grid_size
                else:
                    viewpoints.step(forces, cfg.step_size)

            if key in ("UP", "DOWN", "LEFT", "RIGHT"):
                perturb_active_viewpoint(
                    key, viewpoints, viewer,
                    delta=1.0,              # tweak as you like
                    grid_size=cfg.grid_size
                )
            t3 = time.time()



            # ============================================================
            # Stability analyzer probe (energy, spacing, Jacobian, etc.)
            # ============================================================
            # analyzer.record(
            #     step=step,
            #     positions=viewpoints.positions,
            #     velocities=viewpoints.velocities,
            #     potential=results["potential"],
            #     forces=results["forces"],
            #     need_map=need_map,
            #     force_fn=force_fn  # enables Jacobian + eigen probes
            # )

            # ============================================================
            # Stability logging (spacing, forces, eigenvalues, etc.)
            # ============================================================
            log_step_metrics(
                step=step,
                positions=viewpoints.positions,
                forces=results["forces"],
                potential=results["potential"],
                need_map=need_map,
                cfg=cfg,
                force_fn=force_fn,
                csv_path=CSV_PATH,
                eig_every=1,        # eigen / Jacobian every 50 steps
                jacobian_eps=1e-6,
                print_to_console=True,
            )

            # ------------------------------------------------------------
            # Optional deep-dive diagnostics on selected steps
            # ------------------------------------------------------------
            if step in DEBUG_STEPS:
                debug_step(
                    step=step,
                    sim=sim,
                    cfg=cfg,
                    X=X,
                    Y=Y,
                    positions=viewpoints.positions,
                    need_map=need_map,
                    eps=1e-6,        # match jacobian_eps above
                    force_fn=force_fn,
                )



            # --- visualization ---
            try:
                import cupy as cp
                particles = cp.asnumpy(viewpoints.positions) if isinstance(viewpoints.positions, cp.ndarray) else viewpoints.positions
            except ImportError:
                particles = viewpoints.positions

            # viewer.update({
            #     "particles": particles,
            #     "potential": results["potential"],
            #     "Fx_attr": results["Fx_attr"],
            #     "Fy_attr": results["Fy_attr"],
            #     "Fx_rep": cfg.k_rep * results["Fx_rep"],
            #     "Fy_rep": cfg.k_rep * results["Fy_rep"],
            #     "Fx_total": results["Fx_total"],
            #     "Fy_total": results["Fy_total"],
            #     "F_attr": results["forces_attr"],
            #     "F_rep": results["forces_rep"],
            #     "F_total": results["forces"],
            #     "step": step,
            #     "active_index": viewer.active_index,
            #     "k_rep": cfg.k_rep,
            # })
            viewer.update({
                # Left panel (global potential)
                "particles": particles,
                "potential": results["potential"],

                # Right panel (per-viewpoint vectors)
                "Fx_attr":  Fx_attr_vp,
                "Fy_attr":  Fy_attr_vp,
                "Fx_rep":   Fx_rep_vp,
                "Fy_rep":   Fy_rep_vp,
                "Fx_total": Fx_total_vp,
                "Fy_total": Fy_total_vp,

                # For active viewpoint arrows
                "F_attr": results["forces_attr"],
                "F_rep": results["forces_rep"],
                "F_total": results["forces"],
                "active_index": viewer.active_index,
                "k_rep": cfg.k_rep,
                "need_map": need_map,
            })

            t4 = time.time()

            # --- timing summary ---
            compute_time = t2 - t1
            motion_time = t3 - t2
            viz_time = t4 - t3
            total_time = t4 - t0

            # print(f"[Step {step:04d}] compute={compute_time*1000:6.2f} ms  "
            #     f"motion={motion_time*1000:6.2f} ms  "
            #     f"viz={viz_time*1000:6.2f} ms  total={total_time*1000:6.2f} ms")

except KeyboardInterrupt:
    print("\n[Main] KeyboardInterrupt detected — exiting gracefully.")

finally:
    print("[Main] Closing viewer and saving video (if enabled).")
    try:
        viewer.close()
    except Exception as e:
        print(f"[Main] Viewer close failed: {e}")

    print("[Done] Simulation completed safely.")

