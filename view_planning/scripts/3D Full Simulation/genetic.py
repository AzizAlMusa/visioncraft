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

# ------------------------------- CLI / Utils -------------------------------

def file_stem(path):
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

# ------------------------------- Constants ---------------------------------

TARGET_COVERAGE = 0.9995        # relaxed target coverage
MAX_ROUNDS = 5                 # max times to expand candidates and retry
BASE_NEW_CANDIDATES = 120      # added per retry round
SPHERE_RADIUS = 400            # candidate viewpoint radius
NEAR_PLANE = 300
FAR_PLANE = 900.0
DOWNSAMPLE = 2.0               # for candidate visibility precompute
λ1 = 1.0
λ2 = 2.0

# RKGA params
candidate_density = 5          # initial density factor
generations = 100
pop_size = 50
mutation_rate = 0.2

# ------------------------------- Paths / IO --------------------------------

output_root = "genetic"
output_dir = os.path.join(output_root, model_name, f"seed_{args.seed}")
os.makedirs(output_dir, exist_ok=True)

# ------------------------------- Visualizer --------------------------------

visualizer = Visualizer()
visualizer.initializeWindow("3D View")
visualizer.setBackgroundColor([0.0, 0.0, 0.0])

# --------------------------------- Model -----------------------------------

model = Model()
model.loadModel(model_path, 100000)
visibility_manager = VisibilityManager(model)

# Extract all voxels for metrics (list of tuples from bindings)
all_voxels = list(model.getVoxelMap())
all_voxels_count = len(all_voxels)
all_voxels_set = set(all_voxels)

# ------------------------------ Helpers ------------------------------------

def generate_random_positions(radius, count):
    thetas = np.random.uniform(0, 2 * np.pi, count)
    phis = np.random.uniform(0, np.pi, count)
    positions = []
    for theta, phi in zip(thetas, phis):
        positions.append(np.array([
            radius * np.sin(phi) * np.cos(theta),
            radius * np.sin(phi) * np.sin(theta),
            radius * np.cos(phi)
        ]))
    return positions

def add_candidates(n_more, radius, visibility_sets, candidate_vps, visibility_manager, model):
    new_positions = generate_random_positions(radius, n_more)
    for position in new_positions:
        vp = Viewpoint.from_lookat(position, [0.0, 0.0, 0.0])
        vp.setNearPlane(NEAR_PLANE)
        vp.setFarPlane(FAR_PLANE)
        vp.setDownsampleFactor(DOWNSAMPLE)
        visibility_manager.trackViewpoint(vp)
        vp.performRaycastingOnGPU(model)
        visibility_sets.append(visibility_manager.getVisibleVoxelsForViewpoint(vp))
        visibility_manager.untrackViewpoint(vp)
    candidate_vps.extend(new_positions)

def decode_and_evaluate(chrom, visibility_sets, all_voxels_set, λ1=λ1, λ2=λ2, target=TARGET_COVERAGE):
    sorted_idx = np.argsort(chrom)
    covered = set()
    used = []
    U = len(all_voxels_set)
    goal = int(np.ceil(target * U))

    for idx in sorted_idx:
        idx = int(idx)
        covered.update(visibility_sets[idx])
        used.append(idx)
        if len(covered) >= goal:
            break

    if len(covered) < goal:
        shortfall = goal - len(covered)
        fitness = 1e6 + 1000.0 * shortfall + 10.0 * len(used)
        return fitness, used

    voxel_counts = defaultdict(int)
    for idx in used:
        for v in visibility_sets[idx]:
            voxel_counts[v] += 1
    redundancy = np.array(list(voxel_counts.values()))
    fitness = λ1 * len(used) + λ2 * np.std(redundancy)
    return fitness, used

def rkga_select(visibility_sets, all_voxels_set, generations, pop_size, mutation_rate):
    num_vps = len(visibility_sets)
    pop = np.random.rand(pop_size, num_vps)

    best_chrom = pop[0].copy()
    best_used = []
    best_score = np.inf

    coverage_ts, redundancy_ts, affinity_ts, time_ts = [], [], [], []
    t0 = time.time()
    U = len(all_voxels_set)
    goal = int(np.ceil(TARGET_COVERAGE * max(1, U)))

    for _ in range(generations):
        scores, decoded = [], []
        for c in pop:
            f, used = decode_and_evaluate(c, visibility_sets, all_voxels_set, λ1=λ1, λ2=λ2, target=TARGET_COVERAGE)
            scores.append(f)
            decoded.append(used)

        best_idx = int(np.argmin(scores))
        if scores[best_idx] < best_score:
            best_score = float(scores[best_idx])
            best_chrom = pop[best_idx].copy()
            best_used = list(decoded[best_idx])

        # Build time series from current best
        sorted_idx = np.argsort(best_chrom)
        selected_set = set(best_used)
        covered = set()
        current_selected = []

        for idx in sorted_idx:
            idx = int(idx)
            if idx not in selected_set:
                continue
            current_selected.append(idx)
            covered.update(visibility_sets[idx])

            voxel_counts = defaultdict(int)
            for sel_idx in current_selected:
                for v in visibility_sets[sel_idx]:
                    voxel_counts[v] += 1

            num_covered = len(covered)
            total_hits = sum(voxel_counts.values())
            num_redundant_points = sum(cnt > 1 for cnt in voxel_counts.values())
            redundancy = num_redundant_points / max(1, U)
            affinity = total_hits / max(1, num_covered) if num_covered > 0 else 0.0
            coverage = num_covered / max(1, U)

            coverage_ts.append(coverage)
            redundancy_ts.append(redundancy)
            affinity_ts.append(affinity)
            time_ts.append(time.time() - t0)

            if num_covered >= goal:
                break

        # Evolve population
        parents = pop[np.argsort(scores)[:pop_size // 2]]
        children = []
        for _ in range(pop_size // 2):
            p1 = parents[np.random.randint(len(parents))]
            p2 = parents[np.random.randint(len(parents))]
            alpha = np.random.rand()
            child = alpha * p1 + (1.0 - alpha) * p2
            if np.random.rand() < mutation_rate:
                i = np.random.randint(num_vps)
                child[i] = np.random.rand()
            children.append(child)
        pop = np.vstack([parents, children])

    return best_used, best_chrom, coverage_ts, redundancy_ts, affinity_ts, time_ts

def select_until_target(visibility_sets, candidate_vps, all_voxels_set, sphere_radius,
                        generations, pop_size, mutation_rate, visibility_manager, model):
    def union_frac():
        u = set()
        for s in visibility_sets:
            u |= set(s)
        return len(u) / max(1, len(all_voxels_set))

    round_num = 0
    last_union = 0.0
    final_selected = []
    best_chrom = None
    coverage_ts = []
    redundancy_ts = []
    affinity_ts = []
    time_ts = []

    while round_num < MAX_ROUNDS:
        uf = union_frac()
        print(f"[DEBUG] Round {round_num}: candidate-union coverage = {uf:.4f}")

        selected_idx, best_chrom, c_ts, r_ts, a_ts, t_ts = rkga_select(
            visibility_sets, all_voxels_set, generations, pop_size, mutation_rate
        )

        # Reconstruct in chromosome order
        order = np.argsort(best_chrom)
        goal = int(np.ceil(TARGET_COVERAGE * len(all_voxels_set)))

        final_selected = []
        covered = set()
        sel_set = set(selected_idx)

        for idx in order:
            idx = int(idx)
            if idx not in sel_set:
                continue
            if len(covered) >= goal:
                break
            final_selected.append(idx)
            covered.update(visibility_sets[idx])

        # Minimal deterministic top-up
        if len(covered) < goal:
            remaining = set(range(len(visibility_sets))) - set(final_selected)
            while len(covered) < goal and remaining:
                best_i, best_gain = -1, -1
                for i in list(remaining):
                    gain = len(set(visibility_sets[i]) - covered)
                    if gain > best_gain:
                        best_gain, best_i = gain, i
                if best_gain <= 0:
                    break
                final_selected.append(best_i)
                covered.update(visibility_sets[best_i])
                remaining.remove(best_i)

        achieved = len(covered) / max(1, len(all_voxels_set))
        print(f"[DEBUG] Round {round_num}: achieved coverage = {achieved:.4f}")

        # Carry forward the last TS (no need to concatenate across rounds for files)
        coverage_ts, redundancy_ts, affinity_ts, time_ts = c_ts, r_ts, a_ts, t_ts

        if achieved >= TARGET_COVERAGE:
            return final_selected, best_chrom, coverage_ts, redundancy_ts, affinity_ts, time_ts

        round_num += 1
        add_candidates(BASE_NEW_CANDIDATES, sphere_radius, visibility_sets, candidate_vps, visibility_manager, model)
        if uf <= last_union + 1e-6 and round_num >= MAX_ROUNDS:
            break
        last_union = uf

    return final_selected, best_chrom, coverage_ts, redundancy_ts, affinity_ts, time_ts

# -------------------------- Build initial candidates ------------------------

candidate_vps = generate_random_positions(SPHERE_RADIUS, candidate_density * 20)
visibility_sets = []
for position in candidate_vps:
    vp = Viewpoint.from_lookat(position, [0.0, 0.0, 0.0])
    vp.setNearPlane(NEAR_PLANE)
    vp.setFarPlane(FAR_PLANE)
    vp.setDownsampleFactor(DOWNSAMPLE)
    visibility_manager.trackViewpoint(vp)
    vp.performRaycastingOnGPU(model)
    visibility_sets.append(visibility_manager.getVisibleVoxelsForViewpoint(vp))
    visibility_manager.untrackViewpoint(vp)

# --------------------------- Run selection w/ retries -----------------------

selected_idx, best_chrom, coverage_ts, redundancy_ts, affinity_ts, time_ts = select_until_target(
    visibility_sets, candidate_vps, all_voxels_set, SPHERE_RADIUS, generations, pop_size, mutation_rate,
    visibility_manager, model
)

# ------------------------- Materialize final viewpoints ---------------------

selected_viewpoints = []
for idx in selected_idx:
    pos = candidate_vps[idx]
    vp = Viewpoint.from_lookat(pos, [0.0, 0.0, 0.0])
    vp.setNearPlane(NEAR_PLANE)
    vp.setFarPlane(FAR_PLANE)
    vp.setDownsampleFactor(DOWNSAMPLE)
    visibility_manager.trackViewpoint(vp)
    vp.performRaycastingOnGPU(model)
    selected_viewpoints.append(vp)

print(f"Final Coverage Score: {visibility_manager.getCoverageScore():.4f}")
print(f"Total Viewpoints Selected: {len(selected_viewpoints)}")

if len(selected_viewpoints) == 0:
    print("No viewpoints selected. Skipping save.")
    sys.exit(0)

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

# Save NPZ (matches 2D structure, but assignments are FULL resolution)
npz = os.path.join(output_dir, "genetic_result.npz")
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
csv_path = os.path.join(output_dir, "genetic_viewpoints.csv")
with open(csv_path, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["x", "y", "z", "qx", "qy", "qz", "qw"])
    for vp in selected_viewpoints:
        pos = vp.getPosition()
        quat = vp.getOrientationQuaternion()
        writer.writerow([pos[0], pos[1], pos[2], quat[1], quat[2], quat[3], quat[0]])
print(f"Saved: {csv_path}")

# ------------------------------ Visualization ------------------------------

visualizer.addVoxelMapProperty(model, "visibility", [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
for vp in selected_viewpoints:
    visualizer.addViewpoint(vp, True, True)


# Uncomment to keep the render loop running
print("Press Ctrl+C to exit viewer.")
try:
    while True:
        visualizer.renderStep()
        time.sleep(0.01)
except KeyboardInterrupt:
    print("Exited.")
