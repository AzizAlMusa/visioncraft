#!/usr/bin/env python3
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

output_root = "greedy"
output_dir = os.path.join(output_root, model_name, f"seed_{args.seed}")
os.makedirs(output_dir, exist_ok=True)

visualizer = Visualizer()
visualizer.initializeWindow("3D View")
visualizer.setBackgroundColor([0.0, 0.0, 0.0])

model = Model()
model.loadModel(model_path, 250000)
visibility_manager = VisibilityManager(model)

# Extract all voxels for metrics (list of tuples from bindings)
all_voxels = list(model.getVoxelMap())
all_voxels_count = len(all_voxels)

target_coverage = 0.999
achieved_coverage = visibility_manager.getCoverageScore()
selected_viewpoints = []
coverage_ts = []  # Start empty to match 2D
redundancy_ts = []
affinity_ts = []
time_ts = []
particles = []

start_time = time.time()  # Global start for cumulative time

while achieved_coverage < target_coverage:
    iter_start = time.time()
    best_viewpoint = None
    best_coverage_increment = 0.0
    batch_positions = generate_random_positions(400, 10)

    for position in batch_positions:
        vp = Viewpoint.from_lookat(position, [0.0, 0.0, 0.0])
        vp.setNearPlane(300)
        vp.setFarPlane(900.0)
        vp.setDownsampleFactor(2.0)

        visibility_manager.trackViewpoint(vp)
        vp.performRaycastingOnGPU(model)
        inc = visibility_manager.computeNovelCoverageScore(vp)
        visibility_manager.untrackViewpoint(vp)

        if inc > best_coverage_increment:
            best_coverage_increment = inc
            best_viewpoint = vp

    if best_viewpoint:
        visibility_manager.trackViewpoint(best_viewpoint)
        best_viewpoint.performRaycastingOnGPU(model)

        achieved_coverage = visibility_manager.getCoverageScore()  # Recompute exact
        selected_viewpoints.append(best_viewpoint)
        particles.append(best_viewpoint.getPosition())

        # Compute time-series metrics (exact over all voxels)
        visibility_count = visibility_manager.getVisibilityCount()  # dict: voxel -> count
        num_covered = len(visibility_count)
        total_hits = sum(visibility_count.values())
        num_redundant_points = sum(cnt > 1 for cnt in visibility_count.values())
        redundancy = num_redundant_points / all_voxels_count if all_voxels_count > 0 else 0.0
        affinity = total_hits / num_covered if num_covered > 0 else 0.0

        coverage_ts.append(achieved_coverage)
        redundancy_ts.append(redundancy)
        affinity_ts.append(affinity)
        time_ts.append(time.time() - start_time)  # Cumulative

print(f"Final Coverage Score: {visibility_manager.getCoverageScore():.4f}")
print(f"Total Viewpoints Selected: {len(selected_viewpoints)}")

if len(selected_viewpoints) == 0:
    print("No viewpoints selected. Skipping save.")
    sys.exit(0)

# ---------------- Build visibility map and supporting structures ----------------
visibility_map = visibility_manager.getVisibilityMap()  # dict: vp -> set of voxel tuples
voxel_to_vps = defaultdict(list)
vp_to_index = {vp: i for i, vp in enumerate(selected_viewpoints)}
N = len(selected_viewpoints)

for vp, voxels in visibility_map.items():
    for v in voxels:
        voxel_to_vps[v].append(vp)

# 1) viewpoint_point_assignments: Boolean (V, N) over ALL voxels (no sampling)
viewpoint_point_assignments = np.zeros((len(all_voxels), N), dtype=bool)
for s, voxel in enumerate(all_voxels):
    for vp in voxel_to_vps.get(voxel, []):
        idx = vp_to_index.get(vp)
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
final_viewpoints = np.array(particles)

# ------------------------------ Save NPZ / CSV ------------------------------
npz = os.path.join(output_dir, "greedy_result.npz")
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

csv_path = os.path.join(output_dir, "greedy_viewpoints.csv")
with open(csv_path, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["x", "y", "z", "qx", "qy", "qz", "qw"])
    for vp in selected_viewpoints:
        pos = vp.getPosition()
        quat = vp.getOrientationQuaternion()
        writer.writerow([pos[0], pos[1], pos[2], quat[1], quat[2], quat[3], quat[0]])
print(f"Saved: {csv_path}")

visualizer.addVoxelMapProperty(model, "visibility", [1.0, 1.0, 1.0], [0.0, 1.0, 0.0])
for vp in selected_viewpoints:
    visualizer.addViewpoint(vp, True, True)

# =================== EXACT (NO SAMPLING) VISIBILITY ANALYSIS ===================
# Fresh snapshot of current visibility map for tracked viewpoints
visibility_map_fresh = visibility_manager.getVisibilityMap()

# Map selected viewpoints -> stable indices
vp_to_index_fresh = {vp: i for i, vp in enumerate(selected_viewpoints)}

# Build voxel -> list[vp_idx] using only selected (tracked) viewpoints
voxel_to_indices = defaultdict(list)
for vp, voxels in visibility_map_fresh.items():
    idx = vp_to_index_fresh.get(vp)
    if idx is None:
        continue
    for v in voxels:
        voxel_to_indices[v].append(idx)

all_voxels_count = len(all_voxels)  # already computed above
visible_union = set(voxel_to_indices.keys())
num_visible_union = len(visible_union)
coverage_now = visibility_manager.getCoverageScore()  # should equal num_visible_union / all_voxels_count

# Per-viewpoint totals (exact, over ALL voxels)
total_counts = np.zeros(N, dtype=np.int64)      # |V_j|
exclusive_counts = np.zeros(N, dtype=np.int64)  # |{voxels seen by j and by no other}|

# Redundancy histogram k = number of viewpoints that see a voxel
redundancy_hist = np.zeros(N + 1, dtype=np.int64)  # index k = #viewpoints
sum_k_all_voxels = 0  # sum of k over all voxels (k=0 contributes 0, unseen voxels)

for v, idxs in voxel_to_indices.items():
    k = len(idxs)
    redundancy_hist[k] += 1
    sum_k_all_voxels += k
    # update per-vp totals
    for j in idxs:
        total_counts[j] += 1
    # exclusive (k==1)
    if k == 1:
        exclusive_counts[idxs[0]] += 1

# Voxels unseen by any selected viewpoint:
redundancy_hist[0] = max(all_voxels_count - num_visible_union, 0)

# Fractions over ALL voxels
total_frac_all = total_counts / max(all_voxels_count, 1)
exclusive_frac_all = exclusive_counts / max(all_voxels_count, 1)
nonunique_frac_all = total_frac_all - exclusive_frac_all  # seen by j but also by others

# Mean redundancy over all voxels and over visible voxels
mean_k_all = sum_k_all_voxels / max(all_voxels_count, 1)
mean_k_visible = sum_k_all_voxels / max(num_visible_union, 1)

# Novel frac (all): remove vp_j, evaluate a tracked clone, restore vp_j (re-raycast to repopulate)
novel_frac_all = np.zeros(N, dtype=float)
for j, vp in enumerate(selected_viewpoints):
    # 1) remove original
    visibility_manager.untrackViewpoint(vp)

    tmp = None
    try:
        # 2) evaluate clone at same pose/frustum
        pos = vp.getPosition()
        tmp = Viewpoint.from_lookat(pos, [0.0, 0.0, 0.0])
        tmp.setNearPlane(300)
        tmp.setFarPlane(900.0)
        tmp.setDownsampleFactor(2.0)

        visibility_manager.trackViewpoint(tmp)
        tmp.performRaycastingOnGPU(model)
        novel_frac_all[j] = visibility_manager.computeNovelCoverageScore(tmp)

    finally:
        # Safely untrack tmp if it was created & tracked
        if tmp is not None:
            try:
                visibility_manager.untrackViewpoint(tmp)
            except Exception:
                pass
        # 3) restore original (must re-raycast to repopulate manager's maps for vp)
        visibility_manager.trackViewpoint(vp)
        vp.performRaycastingOnGPU(model)



print("Press Ctrl+C to exit viewer.")
try:
    while True:
        visualizer.renderStep()
        time.sleep(0.01)
except KeyboardInterrupt:
    print("Exited.")
