import numpy as np
import sys
import time
import random
import csv
import os
from collections import defaultdict

sys.path.append("../../build/python_bindings")
from visioncraft_py import Model, Viewpoint, VisibilityManager, Visualizer

import argparse

def file_stem(path):
    """'../../models/gorilla.ply' -> 'gorilla'"""
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    return stem


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="Random seed for the simulation")
parser.add_argument("--model", type=str, default="../../models/gorilla.ply",
                    help="Path to model .ply (e.g., ../../models/gorilla.ply)")
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)

model_path = args.model
model_name = file_stem(model_path)


def file_stem(path):
    """'../../models/gorilla.ply' -> 'gorilla'"""
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    return stem


def generate_random_positions(radius, count):
    return [
        np.array([
            radius * np.sin(phi) * np.cos(theta),
            radius * np.sin(phi) * np.sin(theta),
            radius * np.cos(phi)
        ])
        for theta, phi in zip(
            np.random.uniform(0, 2 * np.pi, count),
            np.random.uniform(0, np.pi, count)
        )
    ]

def sa_perturb_on_sphere(pos, radius, std_ang=np.pi/18):
    # Tangent-plane perturbation + reproject to sphere
    norm_pos = pos / radius
    u = np.cross(norm_pos, [0, 0, 1])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(norm_pos, [1, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(norm_pos, u); v /= np.linalg.norm(v)
    theta = np.random.normal(0, std_ang)
    phi   = np.random.normal(0, std_ang)
    perturbation = np.sin(theta) * u + np.sin(phi) * v
    new_pos = pos + radius * perturbation
    return radius * new_pos / np.linalg.norm(new_pos)

def vis_set_for_position(pos, model, visibility_manager):
    vp = Viewpoint.from_lookat(pos, [0.0, 0.0, 0.0])
    vp.setNearPlane(300)
    vp.setFarPlane(900.0)
    vp.setDownsampleFactor(2.0)
    visibility_manager.trackViewpoint(vp)
    vp.performRaycastingOnGPU(model)
    vs = visibility_manager.getVisibleVoxelsForViewpoint(vp)
    visibility_manager.untrackViewpoint(vp)
    return vs


output_root = "sa"
output_dir = os.path.join(output_root, model_name, f"seed_{args.seed}")
os.makedirs(output_dir, exist_ok=True)



visualizer = Visualizer()
visualizer.initializeWindow("3D View")
visualizer.setBackgroundColor([0.0, 0.0, 0.0])

model = Model()
model.loadModel(model_path, 100000)
visibility_manager = VisibilityManager(model)


# Extract all voxels for metrics (list of tuples from bindings)
all_voxels = list(model.getVoxelMap())
all_voxels_count = len(all_voxels)
all_voxels_set = set(all_voxels)

# Parameters for SA
initial_viewpoints = 6  # Adjust as needed
max_iter = 500
T0 = 0.5
alpha = 0.98
radius = 400.0  # Sphere radius

# Simulated Annealing in 3D, with perturbations on the manifold
def simulated_annealing(
    n_viewpoints=initial_viewpoints,
    max_iter=max_iter,
    target_cov=0.995,
    plateau_window=30,     # steps without improvement to call it a plateau
    plateau_eps=5e-4,      # minimum improvement to count as "better"
    add_cooldown=40,       # wait this many steps after an add before adding again
    N_max=24,              # hard cap on number of viewpoints
    # mini-SA parameters for spawning a new viewpoint
    add_steps=15,
    add_T0=0.3,
    add_alpha=0.9
):
    # --- init positions
    vp = generate_random_positions(radius, n_viewpoints)
    best_vp = vp.copy()

    coverage_ts, redundancy_ts, affinity_ts, time_ts = [], [], [], []
    t0 = time.time()

    # --- precompute vis sets
    visibility_sets = []
    for pos in vp:
        visibility_sets.append(vis_set_for_position(pos, model, visibility_manager))

    # --- objective / metrics
    def evaluate(vis_sets):
        covered = set()
        voxel_counts = defaultdict(int)
        for vs in vis_sets:
            for v in vs:
                voxel_counts[v] += 1
            covered.update(vs)
        cov_val = len(covered) / all_voxels_count
        redundancy_per_point = np.array([voxel_counts[v] for v in covered]) if covered else np.array([])
        obj = 10.0 * (1.0 - cov_val)**2 + 5.0 * np.std(redundancy_per_point) if len(redundancy_per_point) > 0 else np.inf
        num_covered = len(covered)
        total_hits = sum(voxel_counts.values())
        num_redundant_points = sum(cnt > 1 for cnt in voxel_counts.values())
        redundancy = num_redundant_points / all_voxels_count if all_voxels_count > 0 else 0.0
        affinity = total_hits / num_covered if num_covered > 0 else 0.0
        return obj, cov_val, redundancy, affinity

    best_obj, best_cov_val, best_red, best_aff = evaluate(visibility_sets)

    progress_interval = max(1, max_iter // 10)
    step_times = []

    # plateau bookkeeping
    last_improve_step = 0
    last_improve_cov  = best_cov_val
    last_add_step     = -10**9

    for step in range(max_iter):
        step_start = time.time()
        T = T0 * (alpha ** step)

        # ---- SA move on one existing viewpoint (index dynamic because we may add)
        i = np.random.randint(len(best_vp))
        current_pos = best_vp[i]
        new_pos = sa_perturb_on_sphere(current_pos, radius)

        new_vis_set = vis_set_for_position(new_pos, model, visibility_manager)
        new_vis_sets = visibility_sets.copy()
        new_vis_sets[i] = new_vis_set

        new_obj, cov_val, new_red, new_aff = evaluate(new_vis_sets)

        accept = (new_obj < best_obj) or (random.random() < np.exp(-(new_obj - best_obj) / (T + 1e-9)))
        if accept:
            visibility_sets[i] = new_vis_set
            best_vp[i] = new_pos
            improved = (cov_val > best_cov_val + plateau_eps)
            if improved:
                best_cov_val = cov_val
                best_obj = new_obj
                best_red = new_red
                best_aff = new_aff
                last_improve_step = step
                last_improve_cov  = best_cov_val
            else:
                best_obj = new_obj
                best_red = new_red
                best_aff = new_aff

        # ---- time series
        coverage_ts.append(best_cov_val)
        redundancy_ts.append(best_red)
        affinity_ts.append(best_aff)
        time_ts.append(time.time() - t0)

        # ---- plateau check -> spawn a new viewpoint with a tiny SA (no greedy batch)
        plateau_elapsed = step - last_improve_step
        since_last_add  = step - last_add_step
        if (best_cov_val < target_cov and
            len(best_vp) < N_max and
            plateau_elapsed >= plateau_window and
            since_last_add >= add_cooldown):

            # mini-SA for a single new viewpoint while others are fixed
            cand_pos = generate_random_positions(radius, 1)[0]
            cand_vs  = vis_set_for_position(cand_pos, model, visibility_manager)
            # baseline with candidate included
            base_sets = visibility_sets + [cand_vs]
            cand_obj, cand_cov, cand_red, cand_aff = evaluate(base_sets)
            cand_best = (cand_obj, cand_cov, cand_red, cand_aff, cand_pos, cand_vs)

            T_add = add_T0
            for _ in range(add_steps):
                trial_pos = sa_perturb_on_sphere(cand_pos, radius)
                trial_vs  = vis_set_for_position(trial_pos, model, visibility_manager)
                trial_sets = visibility_sets + [trial_vs]
                trial_obj, trial_cov, trial_red, trial_aff = evaluate(trial_sets)

                dE = trial_obj - cand_obj
                if (trial_obj < cand_obj) or (random.random() < np.exp(-dE / (T_add + 1e-12))):
                    cand_pos, cand_vs = trial_pos, trial_vs
                    cand_obj, cand_cov, cand_red, cand_aff = trial_obj, trial_cov, trial_red, trial_aff
                    if cand_cov > cand_best[1] + 1e-12:
                        cand_best = (cand_obj, cand_cov, cand_red, cand_aff, cand_pos, cand_vs)
                T_add *= add_alpha

            # commit the best candidate from the mini-SA
            _, add_cov, add_red, add_aff, add_pos, add_vs = cand_best
            visibility_sets.append(add_vs)
            best_vp.append(add_pos)
            best_obj, best_cov_val, best_red, best_aff = evaluate(visibility_sets)
            last_add_step = step
            last_improve_step = step
            last_improve_cov  = best_cov_val
            print(f"[ADD-SA] +1 viewpoint (N={len(best_vp)}). Coverage={best_cov_val:.4f}")

        # ---- progress + early stop
        step_time = time.time() - step_start
        step_times.append(step_time)
        if step % progress_interval == 0 and step > 0:
            avg_step_time = np.mean(step_times)
            remaining_steps = max_iter - step
            eta = remaining_steps * avg_step_time / 60
            print(f"Progress: {step}/{max_iter} ({(step / max_iter * 100):.1f}%) | "
                  f"Cov: {best_cov_val:.4f} | Obj: {best_obj:.4f} | N={len(best_vp)} | ETA: {eta:.2f} min")

        if best_cov_val >= target_cov:
            print(f"[Early Stop] Coverage >= {target_cov} at step {step} with N={len(best_vp)}")
            break

    visibility_manager.untrackAllViewpoints()
    return best_vp, coverage_ts, redundancy_ts, affinity_ts, time_ts

# Run SA
viewpoints, coverage_ts, redundancy_ts, affinity_ts, time_ts = simulated_annealing()

# Track final viewpoints for metrics and visualization
selected_viewpoints = []
for pos in viewpoints:
    vp = Viewpoint.from_lookat(pos, [0.0, 0.0, 0.0])
    vp.setNearPlane(300)
    vp.setFarPlane(900.0)
    vp.setDownsampleFactor(2.0)
    visibility_manager.trackViewpoint(vp)
    vp.performRaycastingOnGPU(model)
    selected_viewpoints.append(vp)

print(f"Final Coverage Score: {visibility_manager.getCoverageScore():.4f}")
print(f"Total Viewpoints Selected: {len(selected_viewpoints)}")

if len(selected_viewpoints) == 0:
    print("No viewpoints selected. Skipping save.")
    exit(0)

# ------------------------------ Metrics export (NO SAMPLING) ------------------------------

# Build visibility map and supporting structures
visibility_map = visibility_manager.getVisibilityMap()  # dict: vp -> set of voxel tuples
voxel_to_vps = defaultdict(list)
vp_index_map = {vp: i for i, vp in enumerate(selected_viewpoints)}

for vp, voxels in visibility_map.items():
    for v in voxels:
        voxel_to_vps[v].append(vp)

# 1) viewpoint_point_assignments: Boolean (V, N) over ALL voxels (no sampling)
N = len(selected_viewpoints)
viewpoint_point_assignments = np.zeros((len(all_voxels), N), dtype=bool)
for s, voxel in enumerate(all_voxels):
    for vp in voxel_to_vps.get(voxel, []):
        idx = vp_index_map.get(vp)
        if idx is not None:
            viewpoint_point_assignments[s, idx] = True

# 2) viewpoint_contribution_hist: #points seen by each viewpoint (over ALL voxels)
viewpoint_contribution_hist = np.sum(viewpoint_point_assignments, axis=0).astype(np.int32)

# 3) point_redundancy_hist: Histogram #VPs per point (over ALL voxels, includes 0s)
redundancy_counts = np.sum(viewpoint_point_assignments, axis=1)
point_redundancy_hist = np.bincount(
    redundancy_counts,
    minlength=redundancy_counts.max() + 1 if len(redundancy_counts) > 0 else 1
).astype(np.int32)

# 4) viewpoint_overlap_matrix: N×N Jaccard overlap (over ALL voxels)
overlap = np.zeros((N, N), dtype=np.float32)
epsilon = 1e-6
for i in range(N):
    Vi = viewpoint_point_assignments[:, i]
    for j in range(i + 1, N):
        Vj = viewpoint_point_assignments[:, j]
        inter = np.logical_and(Vi, Vj).sum()
        union = np.logical_or(Vi, Vj).sum()
        jac = inter / (union + epsilon)
        overlap[i, j] = overlap[j, i] = jac
np.fill_diagonal(overlap, 1.0)

# 5) functional_isolation_flags: True if mean overlap <5%
if N > 0:
    avg_ov = (np.sum(overlap, axis=1) - 1) / max(N - 1, 1)
    functional_isolation_flags = (avg_ov < 0.05)
else:
    functional_isolation_flags = np.array([], dtype=bool)

# final_viewpoints
final_viewpoints = np.array([vp.getPosition() for vp in selected_viewpoints])

# Save NPZ (matches 2D structure; assignments are full-resolution)
npz = os.path.join(output_dir, "sa_result.npz")
np.savez_compressed(
    npz,
    coverage=np.array(coverage_ts),
    redundancy=np.array(redundancy_ts),
    affinity=np.array(affinity_ts),
    time=np.array(time_ts),
    num_viewpoints=N,
    num_iterations=len(coverage_ts),
    viewpoint_point_assignments=viewpoint_point_assignments,
    viewpoint_contribution_hist=viewpoint_contribution_hist,
    point_redundancy_hist=point_redundancy_hist,
    viewpoint_overlap_matrix=overlap,
    functional_isolation_flags=functional_isolation_flags,
    final_viewpoints=final_viewpoints
)
print(f"[Saved] Metrics NPZ ➜ {npz}")

# Save CSV
csv_path = os.path.join(output_dir, "sa_viewpoints.csv")
with open(csv_path, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["x", "y", "z", "qx", "qy", "qz", "qw"])
    for vp in selected_viewpoints:
        pos = vp.getPosition()
        quat = vp.getOrientationQuaternion()
        writer.writerow([pos[0], pos[1], pos[2], quat[1], quat[2], quat[3], quat[0]])
print(f"Saved: {csv_path}")

# Visualization
visualizer.addVoxelMapProperty(model, "visibility", [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])  # Red unseen, green seen
for vp in selected_viewpoints:
    visualizer.addViewpoint(vp, True, True)

# print("Press Ctrl+C to exit viewer.")
# try:
#     while True:
#         visualizer.renderStep()
#         time.sleep(0.01)
# except KeyboardInterrupt:
#     print("Exited.")