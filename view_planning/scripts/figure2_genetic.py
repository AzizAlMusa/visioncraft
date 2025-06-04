#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse, os, time

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--strategy", type=str, default="rkga", choices=["rkga"])
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
max_viewpoints = 100
x, y = np.arange(grid_size), np.arange(grid_size)
X, Y = np.meshgrid(x, y)
field_points = np.stack([X.ravel(), Y.ravel()], axis=-1)

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

# ---------- Candidate Generation ----------
def generate_candidates(n):
    idx = np.random.choice(len(field_points), size=n, replace=False)
    noise = np.random.normal(0, fov_radius * 0.3, (n, 2))
    return (field_points[idx] + noise) % grid_size

# ---------- Decoder + Fitness ----------
def decode_and_evaluate(chrom, A, cov_thresh=0.25, λ1=1.0, λ2=2.0):
    sorted_idx = np.argsort(chrom)
    covered = np.zeros(A.shape[0], dtype=bool)
    used = []
    for idx in sorted_idx:
        covered |= (A[:, idx] >= cov_thresh)
        used.append(idx)
        if covered.all(): break
    if not covered.all(): return np.inf, []
    redundancy = (A[:, used] >= cov_thresh).sum(axis=1)
    fitness = λ1 * len(used) + λ2 * np.std(redundancy)
    return fitness, used

# ---------- RKGA Loop ----------
def rkga_select(A, cov_thresh=0.25, generations=100, pop_size=50, mutation_rate=0.2):
    num_vps = A.shape[1]
    pop = np.random.rand(pop_size, num_vps)
    best_chrom, best_used, best_score = None, None, np.inf

    coverage_ts, redundancy_ts, affinity_ts, time_ts = [], [], [], []
    t0 = time.time()

    for gen in range(generations):
        scores, decoded = [], []
        for c in pop:
            f, used = decode_and_evaluate(c, A, cov_thresh)
            scores.append(f)
            decoded.append(used)

        best_idx = np.argmin(scores)
        if scores[best_idx] < best_score:
            best_score = scores[best_idx]
            best_chrom = pop[best_idx].copy()
            best_used = decoded[best_idx]

        selected = decoded[best_idx]
        sorted_idx = np.argsort(best_chrom)
        selected_set = set(selected)
        covered = np.zeros(A.shape[0], dtype=bool)
        current_selected = []

        for idx in sorted_idx:
            if idx not in selected_set:
                continue
            current_selected.append(idx)
            covered |= (A[:, idx] >= cov_thresh)

            mask = A[:, current_selected] >= cov_thresh
            redundancy = np.mean(mask.sum(axis=1) > 1)
            v = mask.sum(axis=1)
            affinity = np.mean(v[v > 0]) if np.any(v > 0) else 0.0
            coverage = np.mean(mask.any(axis=1))

            coverage_ts.append(coverage)
            redundancy_ts.append(redundancy)
            affinity_ts.append(affinity)
            time_ts.append(time.time() - t0)

            if coverage >= 1.0:
                break

        parents = pop[np.argsort(scores)[:pop_size//2]]
        children = []
        for _ in range(pop_size//2):
            p1, p2 = parents[np.random.randint(len(parents))], parents[np.random.randint(len(parents))]
            alpha = np.random.rand()
            child = alpha * p1 + (1 - alpha) * p2
            if np.random.rand() < mutation_rate:
                i = np.random.randint(num_vps)
                child[i] = np.random.rand()
            children.append(child)
        pop = np.vstack([parents, children])

    return best_used, best_chrom, coverage_ts, redundancy_ts, affinity_ts, time_ts

# ---------- Main ----------
os.makedirs(args.save_dir, exist_ok=True)
candidate_vps = generate_candidates(args.candidate_density * 20)
visibility_matrix = np.array([compute_visibility(vp) for vp in candidate_vps]).T

selected_idx, best_chrom, coverage_ts, redundancy_ts, affinity_ts, time_ts = rkga_select(
    visibility_matrix, cov_thresh=args.coverage_threshold, generations=100
)
selected_idx_sorted = np.argsort(best_chrom)

# ---------- Final Reconstruction ----------
final_selected = []
covered = np.zeros(visibility_matrix.shape[0], dtype=bool)
for idx in selected_idx_sorted:
    covered |= (visibility_matrix[:, idx] >= args.coverage_threshold)
    final_selected.append(idx)
    if covered.all(): break
selected_idx = final_selected
viewpoints = candidate_vps[selected_idx]

# ---------- Snapshots / animation ----------
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
    print(f"[Saved] Animation  ➔ {mp4}")

# ---------- Final metrics ----------
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
        Vj  = vis[:,j]
        inter = np.logical_and(Vi,Vj).sum()
        union = np.logical_or (Vi,Vj).sum()
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
print(f"[Saved] Metrics NPZ ➔ {npz}")
print("[Done] RKGA-based SCP simulation complete.")