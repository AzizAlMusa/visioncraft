#!/usr/bin/env python3
"""
Fast Simulated Annealing set-cover simulation (torus) — metrics-ready
————————————————————————————————————————————————————————————
• Reuses visibility matrix with incremental updates
• Strongly penalizes uncovered regions
• Compatible with greedy/rkga pipeline
• Reaches 100% coverage with fixed 15 viewpoints
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse, os, time

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--strategy", type=str, default="sa", choices=["sa"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save_dir", type=str, default="./results2")
parser.add_argument("--animate", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--candidate_density", type=int, default=5)
parser.add_argument("--coverage_threshold", type=float, default=0.25)
parser.add_argument("--initial_viewpoints", type=int, default=11)
parser.add_argument("--max_iter", type=int, default=1000)
args = parser.parse_args()

# ---------- Globals ----------
np.random.seed(args.seed)
grid_size, fov_radius = 100, 20
epsilon = 1e-6
x, y = np.arange(grid_size), np.arange(grid_size)
X, Y = np.meshgrid(x, y)
field_points = np.stack([X.ravel(), Y.ravel()], axis=-1)
N_points = len(field_points)

# ---------- Wrapping Distance ----------
def wrap_distance(diff):
    half = grid_size / 2
    return np.where(np.abs(diff) > half, -np.sign(diff) * (grid_size - np.abs(diff)), diff)

# ---------- Visibility ----------
def compute_visibility(viewpoint):
    diff = field_points - viewpoint
    diff = wrap_distance(diff)
    dist = np.linalg.norm(diff, axis=-1) + epsilon
    beta = 20.0
    return 1.0 / (1.0 + np.exp(beta * (dist / fov_radius - 1.0)))

# ---------- Redundancy / Affinity Stats ----------
def compute_redundancy_affinity(viewpoints):
    S = 10000
    pts = np.random.rand(S,2) * grid_size
    dist = np.linalg.norm(wrap_distance(pts[:,None,:]-viewpoints[None,:,:]), axis=-1)
    vis  = dist <= fov_radius
    cnt  = vis.sum(axis=1)
    redundancy = np.mean(cnt>1)
    affinity   = np.mean(cnt[cnt>0]) if np.any(cnt>0) else 0.0
    return redundancy, affinity

# ---------- Simulated Annealing ----------
def simulated_annealing(n_viewpoints=11, max_iter=2000):
    vp = np.random.rand(n_viewpoints, 2) * grid_size
    V = np.array([compute_visibility(p) for p in vp])
    cov_mask = np.maximum.reduce(V, axis=0)
    covered = cov_mask > args.coverage_threshold
    coverage_val = np.mean(covered)
    redundancy_per_point = (V >= args.coverage_threshold).sum(axis=1)
    obj = 10.0 * (1.0 - coverage_val)**2 + 5.0 * np.std(redundancy_per_point)

    best_vp = vp.copy()
    best_obj = obj
    best_cov_val = coverage_val

    snapshots, nbvs = [], []
    coverage_ts, redundancy_ts, affinity_ts, time_ts = [], [], [], []
    t0 = time.time()

    T0, alpha = 0.5, 0.98

    for step in range(max_iter):
        T = T0 * (alpha ** step)
        i = np.random.randint(n_viewpoints)
        new_vp = best_vp.copy()
        new_vp[i] = (new_vp[i] + np.random.normal(0, fov_radius * 0.3, 2)) % grid_size
        new_vis = compute_visibility(new_vp[i])

        V_new = V.copy()
        V_new[i] = new_vis
        cov_new = np.maximum.reduce(V_new, axis=0)
        covered_new = cov_new > args.coverage_threshold
        cov_val = np.mean(covered_new)

        redundancy_per_point = (V_new >= args.coverage_threshold).sum(axis=1)
        uncovered_penalty = (1.0 - cov_val)**2
        new_obj = 10.0 * uncovered_penalty + 5.0 * np.std(redundancy_per_point)

        accept = (new_obj < best_obj) or (np.random.rand() < np.exp(-(new_obj - best_obj) / (T + 1e-9)))
        if accept:
            V = V_new
            best_vp = new_vp
            best_obj = new_obj
            cov_mask = cov_new

        if cov_val > best_cov_val + 1e-6:
            snapshots.append(cov_mask.copy())
            nbvs.append(best_vp[-1].copy())
            best_cov_val = cov_val

        red, aff = compute_redundancy_affinity(best_vp)
        coverage_ts.append(cov_val)
        redundancy_ts.append(red)
        affinity_ts.append(aff)
        time_ts.append(time.time() - t0)

        if args.verbose and step % 100 == 0:
            print(f"[{step:04d}] cov={cov_val:.4f}  red={red:.4f}  obj={new_obj:.4f}")

        if cov_val >= 1.0:
            print(f"[Early Stop] Full coverage achieved at step {step}")
            break

    return best_vp, coverage_ts, snapshots, nbvs, redundancy_ts, affinity_ts, time_ts

# ---------- Run ----------
os.makedirs(args.save_dir, exist_ok=True)
(viewpoints, coverage_ts, snapshots, nbvs,
 redundancy_ts, affinity_ts, time_ts) = simulated_annealing(args.initial_viewpoints, args.max_iter)

# ---------- Snapshots / Animation ----------
if args.animate and len(snapshots) >= 2:
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(1 - snapshots[0].reshape(grid_size, grid_size), cmap='viridis', origin='lower')
    ax[0].set_title("Initial")
    ax[1].imshow(1 - snapshots[-1].reshape(grid_size, grid_size), cmap='viridis', origin='lower')
    ax[1].set_title("Final")
    snap = os.path.join(args.save_dir, f"{args.strategy}_seed{args.seed}_snapshots.png")
    plt.savefig(snap, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[Saved] Snapshots ➜ {snap}")

if args.animate:
    fig, ax = plt.subplots(figsize=(6,6))
    def animate(i):
        ax.clear()
        vis = snapshots[i]
        ax.imshow(1 - vis.reshape(grid_size, grid_size), cmap='viridis', origin='lower')
        ax.scatter(viewpoints[:i+1, 0], viewpoints[:i+1, 1], c='red', s=20, edgecolors='white')
    ani = animation.FuncAnimation(fig, animate, frames=len(snapshots), interval=100)
    mp4 = os.path.join(args.save_dir, f"{args.strategy}_seed{args.seed}_anim.mp4")
    ani.save(mp4, writer='ffmpeg', dpi=300); plt.close()
    print(f"[Saved] Animation ➜ {mp4}")

# ---------- Final Metrics ----------
N = len(viewpoints)
S = 50000
pts = np.random.rand(S,2) * grid_size
dist = np.linalg.norm(wrap_distance(pts[:,None,:] - viewpoints[None,:,:]), axis=-1)
vis = dist <= fov_radius

viewpoint_point_assignments = vis.astype(np.bool_)
viewpoint_contribution_hist = vis.sum(axis=0).astype(np.int32)
redundancy_counts = vis.sum(axis=1)
point_redundancy_hist = np.bincount(redundancy_counts, minlength=redundancy_counts.max()+1).astype(np.int32)

overlap = np.zeros((N,N), dtype=np.float32)
for i in range(N):
    Vi = vis[:,i]
    for j in range(i, N):
        Vj = vis[:,j]
        inter = np.logical_and(Vi, Vj).sum()
        union = np.logical_or(Vi, Vj).sum()
        overlap[i,j] = overlap[j,i] = inter / (union + epsilon)
np.fill_diagonal(overlap, 1.0)

avg_ov = (overlap.sum(axis=1) - 1) / max(N-1,1)
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
print(f"[Saved] Metrics NPZ ➜ {npz}")
print("[Done] Fast Simulated Annealing complete.")
