#!/usr/bin/env python3
"""
PSO-based set cover simulation (torus) — metrics-ready version
————————————————————————————————————————————————————————————
• Fully compatible CLI and behavior with GREEDY / GENETIC.
• Automatically searches for number of viewpoints (not fixed).
• Outputs identical NPZ structure with all final configuration data.
• Supports animation and snapshot generation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse, os, time

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--strategy", type=str, default="pso", choices=["pso"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save_dir", type=str, default="./results2")
parser.add_argument("--animate", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--candidate_density", type=int, default=5)
parser.add_argument("--coverage_threshold", type=float, default=0.25)
args = parser.parse_args()

# ---------- Globals ----------
np.random.seed(args.seed)
grid_size, fov_radius = 100, 20
epsilon = 1e-6
x, y = np.arange(grid_size), np.arange(grid_size)
X, Y = np.meshgrid(x, y)
field_points = np.stack([X.ravel(), Y.ravel()], axis=-1)
max_particles = 60
max_iters = 100
λ1, λ2, λ3 = 1.0, 2.0, 0.02
ideal_height = 100  # not used but from paper

# ---------- Visibility ----------
def wrap_distance(diff):
    half = grid_size / 2
    return np.where(np.abs(diff) > half, -np.sign(diff) * (grid_size - np.abs(diff)), diff)

def compute_visibility(viewpoint):
    diff = field_points - viewpoint
    diff = wrap_distance(diff)
    dist = np.linalg.norm(diff, axis=-1) + epsilon
    beta = 20.0
    return 1.0 / (1.0 + np.exp(beta * (dist / fov_radius - 1.0)))

# ---------- Generate Candidate VPs ----------
def generate_candidates(n):
    idx = np.random.choice(len(field_points), size=n, replace=False)
    noise = np.random.normal(0, fov_radius * 0.3, (n, 2))
    return (field_points[idx] + noise) % grid_size

# ---------- Decode and Fitness ----------
def decode_and_evaluate(chrom, A, cov_thresh=0.25):
    sorted_idx = np.argsort(chrom)
    covered = np.zeros(A.shape[0], dtype=bool)
    used = []
    for idx in sorted_idx:
        covered |= (A[:, idx] >= cov_thresh)
        used.append(idx)
        if covered.all(): break
    if not covered.all(): return np.inf, []

    mask = A[:, used] >= cov_thresh
    redundancy = mask.sum(axis=1)
    redundancy_std = np.std(redundancy)
    redundant_area = np.sum(redundancy > 1)
    fitness = λ1 * len(used) + λ2 * redundancy_std + λ3 * redundant_area
    return fitness, used

# ---------- Adaptive Inertia PSO ----------
def pso_select(A, cov_thresh=0.25, max_iters=100):
    num_vps = A.shape[1]
    pop = np.random.rand(max_particles, num_vps)
    vel = np.random.rand(max_particles, num_vps) * 0.1
    p_best = pop.copy()
    p_best_scores = np.full(max_particles, np.inf)
    g_best = None
    g_best_score = np.inf

    coverage_ts, redundancy_ts, affinity_ts, time_ts = [], [], [], []
    t0 = time.time()

    for k in range(max_iters):
        w = 0.4 + 0.5 * (1 - (k / max_iters)) ** (1/3)  # nonlinear adaptive inertia
        for i in range(max_particles):
            f, used = decode_and_evaluate(pop[i], A, cov_thresh)
            if f < p_best_scores[i]:
                p_best_scores[i] = f
                p_best[i] = pop[i].copy()
            if f < g_best_score:
                g_best_score = f
                g_best = pop[i].copy()
                g_best_used = used.copy()

        for i in range(max_particles):
            r1, r2 = np.random.rand(num_vps), np.random.rand(num_vps)
            vel[i] = w * vel[i] \
                   + 1.4 * r1 * (p_best[i] - pop[i]) \
                   + 1.4 * r2 * (g_best - pop[i])
            pop[i] += vel[i]

        # record g_best trace
        mask = A[:, g_best_used] >= cov_thresh
        redundancy = np.mean(mask.sum(axis=1) > 1)
        v = mask.sum(axis=1)
        affinity = np.mean(v[v > 0]) if np.any(v > 0) else 0.0
        coverage = np.mean(mask.any(axis=1))
        coverage_ts.append(coverage)
        redundancy_ts.append(redundancy)
        affinity_ts.append(affinity)
        time_ts.append(time.time() - t0)
        if args.verbose:
            print(f"[{k:03d}] coverage={coverage:.3f}  viewpoints={len(g_best_used):d}")

    return g_best_used, g_best, coverage_ts, redundancy_ts, affinity_ts, time_ts

# ---------- Main ----------
os.makedirs(args.save_dir, exist_ok=True)
candidate_vps = generate_candidates(args.candidate_density * 20)
visibility_matrix = np.array([compute_visibility(vp) for vp in candidate_vps]).T

selected_idx, best_chrom, coverage_ts, redundancy_ts, affinity_ts, time_ts = pso_select(
    visibility_matrix, cov_thresh=args.coverage_threshold, max_iters=max_iters
)
viewpoints = candidate_vps[selected_idx]

# ---------- Snapshots ----------
snapshots = [np.maximum.reduce(visibility_matrix[:, selected_idx[:i+1]], axis=1)
             for i in range(len(selected_idx))]

if args.animate and len(snapshots) >= 2:
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(1 - snapshots[0].reshape(grid_size, grid_size), cmap='viridis', origin='lower')
    ax[0].set_title("Initial")
    ax[1].imshow(1 - snapshots[-1].reshape(grid_size, grid_size), cmap='viridis', origin='lower')
    ax[1].set_title("Final")
    snap = os.path.join(args.save_dir, f"{args.strategy}_seed{args.seed}_snapshots.png")
    plt.savefig(snap, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[Saved] Snapshots ➔ {snap}")

if args.animate:
    fig, ax = plt.subplots(figsize=(6, 6))
    def animate(i):
        ax.clear()
        vis = snapshots[i]
        ax.imshow(1 - vis.reshape(grid_size, grid_size), cmap='viridis', origin='lower')
        ax.scatter(viewpoints[:i+1, 0], viewpoints[:i+1, 1], c='red', s=20, edgecolors='white')
    ani = animation.FuncAnimation(fig, animate, frames=len(snapshots), interval=100)
    mp4 = os.path.join(args.save_dir, f"{args.strategy}_seed{args.seed}_anim.mp4")
    ani.save(mp4, writer='ffmpeg', dpi=300); plt.close()
    print(f"[Saved] Animation ➔ {mp4}")

# ---------- Final metrics ----------
N = len(viewpoints)
S = 50000
pts = np.random.rand(S, 2) * grid_size
dist = np.linalg.norm(wrap_distance(pts[:, None, :] - viewpoints[None, :, :]), axis=-1)
vis = dist <= fov_radius

viewpoint_point_assignments = vis.astype(np.bool_)
viewpoint_contribution_hist = vis.sum(axis=0).astype(np.int32)
redundancy_counts = vis.sum(axis=1)
point_redundancy_hist = np.bincount(redundancy_counts, minlength=redundancy_counts.max()+1).astype(np.int32)

overlap = np.zeros((N, N), dtype=np.float32)
for i in range(N):
    Vi = vis[:, i]
    for j in range(i, N):
        Vj = vis[:, j]
        inter = np.logical_and(Vi, Vj).sum()
        union = np.logical_or(Vi, Vj).sum()
        overlap[i, j] = overlap[j, i] = inter / (union + epsilon)
np.fill_diagonal(overlap, 1.0)

avg_ov = (overlap.sum(axis=1) - 1) / max(N - 1, 1)
functional_isolation_flags = (avg_ov < 0.05)

# ---------- Save ----------
npz = os.path.join(args.save_dir, f"{args.strategy}_seed{args.seed}_metrics.npz")
np.savez_compressed(npz,
    coverage       = np.array(coverage_ts),
    redundancy     = np.array(redundancy_ts),
    affinity       = np.array(affinity_ts),
    time           = np.array(time_ts),
    num_viewpoints = N,
    num_iterations = len(coverage_ts),
    viewpoint_point_assignments = viewpoint_point_assignments,
    viewpoint_contribution_hist = viewpoint_contribution_hist,
    point_redundancy_hist       = point_redundancy_hist,
    viewpoint_overlap_matrix    = overlap,
    functional_isolation_flags  = functional_isolation_flags
)
print(f"[Saved] Metrics NPZ ➔ {npz}")
print("[Done] PSO-based SCP simulation complete.")
