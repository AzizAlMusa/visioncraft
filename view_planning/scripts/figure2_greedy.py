#!/usr/bin/env python3
"""
Greedy set-cover simulation (torus)  —  metrics-ready version
————————————————————————————————————————————————————————————
• Keeps **all** previous CLI flags, filenames, snapshots, and animation behaviour.
• Drops the post-processing scalars you no longer want
      (marginal_gain, stray_gap_views, avg_overlap_degree,
       num_isolated_viewpoints, uniformity_score).
• Adds final-configuration data, identical in structure to the potential-field
  script delivered earlier:

      viewpoint_point_assignments   ⟹ Boolean (S_points , N_viewpoints)
      viewpoint_contribution_hist   ⟹ #points seen by each viewpoint
      point_redundancy_hist         ⟹ histogram: #VPs per point
      viewpoint_overlap_matrix      ⟹ N×N Jaccard overlap between VPs
      functional_isolation_flags    ⟹ True if a VP’s mean overlap < 5 %

The standard time-series arrays (coverage, redundancy, affinity, time) remain
unchanged, so downstream analysis scripts stay compatible.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse, os, time

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--strategy", type=str, default="greedy", choices=["greedy"])
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
epsilon               = 1e-6
max_viewpoints        = 100
x, y                  = np.arange(grid_size), np.arange(grid_size)
X, Y                  = np.meshgrid(x, y)
field_points          = np.stack([X.ravel(), Y.ravel()], axis=-1)

# ---------- Greedy coverage class ----------
class GreedyCoverage:
    def __init__(self, grid_size, fov_radius, use_wrapping=True):
        self.grid_size = grid_size
        self.fov_radius = fov_radius
        self.use_wrapping = use_wrapping
        self.coverage_threshold = args.coverage_threshold
        self.field_points = field_points                                   # (N²,2)

    def wrap_distance(self, diff):
        if not self.use_wrapping: return diff
        half = self.grid_size / 2
        return np.where(np.abs(diff) > half,
                        -np.sign(diff) * (self.grid_size - np.abs(diff)), diff)

    # ---- single-view visibility sigmoid ----
    def compute_visibility(self, viewpoint):
        diff = self.field_points - viewpoint
        diff = self.wrap_distance(diff)
        dist = np.linalg.norm(diff, axis=-1) + epsilon
        beta = 20.0
        return 1.0 / (1.0 + np.exp(beta * (dist / self.fov_radius - 1.0)))

    # ---- helpers for redundancy / affinity (Monte-Carlo) ----
    def _mc_stats(self, viewpoints, S=10000):
        pts = np.random.rand(S,2) * self.grid_size
        dist = np.linalg.norm(self.wrap_distance(pts[:,None,:]-viewpoints[None,:,:]), axis=-1)
        vis  = dist <= self.fov_radius
        cnt  = vis.sum(axis=1)
        redundancy = np.mean(cnt>1)
        affinity   = np.mean(cnt[cnt>0]) if np.any(cnt>0) else 0.0
        return redundancy, affinity

    # ---- candidates centred on low-coverage regions ----
    def generate_candidates(self, n):
        cov = self.current_coverage
        weights = 1.0 - cov + epsilon
        weights /= weights.sum()
        idx = np.random.choice(len(self.field_points), size=n, replace=False, p=weights)
        noise = np.random.normal(0, self.fov_radius*0.3, (n,2))
        return (self.field_points[idx] + noise) % self.grid_size

    # ---- main greedy loop ----
    def run(self, max_vp=100, target=1.0):
        self.current_coverage = np.zeros(len(self.field_points))
        vp = []
        coverage_ts, redundancy_ts, affinity_ts, time_ts = [], [], [], []
        snapshots, nbvs = [], []
        t0 = time.time()

        while len(vp) < max_vp:
            covered = np.mean(self.current_coverage > self.coverage_threshold)
            if covered >= target: break

            best_gain, best_vp, best_vis = -1, None, None
            for cand in self.generate_candidates(args.candidate_density*20):
                vis = self.compute_visibility(cand)
                new_cov = np.maximum(self.current_coverage, vis)
                new_pts = np.sum((new_cov>self.coverage_threshold) &
                                 (self.current_coverage<=self.coverage_threshold))
                score = new_pts + 0.1 * new_cov[new_cov>self.coverage_threshold].mean()
                if score > best_gain:
                    best_gain, best_vp, best_vis = score, cand, vis

            if best_gain <= 0: break       # stuck
            vp.append(best_vp)
            self.current_coverage = np.maximum(self.current_coverage, best_vis)

            snapshots.append(self.current_coverage.copy())
            nbvs.append(best_vp.copy())

            cov_now = np.mean(self.current_coverage > self.coverage_threshold)
            red, aff = self._mc_stats(np.array(vp))
            coverage_ts.append(cov_now); redundancy_ts.append(red); affinity_ts.append(aff)
            time_ts.append(time.time() - t0)

            if args.verbose:
                print(f"[{len(vp):03d}] cov={cov_now:6.3f} red={red:5.3f} vp={len(vp):3d}")

        return (np.array(vp), np.array(coverage_ts), np.array(snapshots),
                np.array(nbvs), np.array(redundancy_ts), np.array(affinity_ts),
                np.array(time_ts))

# ---------- Plot helper ----------
def plot_frame(ax, pts, pot, nbv=None, title=None):
    ax.clear()
    im = ax.imshow(pot, cmap='viridis', origin='lower',
                   extent=[0, grid_size, 0, grid_size],
                   vmin=pot.min(), vmax=pot.max())
    for p in pts:
        for dx in (-grid_size, 0, grid_size):
            for dy in (-grid_size, 0, grid_size):
                ax.add_patch(plt.Circle((p[0]+dx, p[1]+dy), fov_radius,
                              edgecolor='white', facecolor='none', alpha=0.2, lw=1))
    ax.scatter(pts[:, 0], pts[:, 1], c='deepskyblue', s=30,
               edgecolors='white', linewidths=0.6)
    if nbv is not None:
        ax.scatter(*nbv, s=60, c='red', edgecolors='white', lw=1.5)
        ax.text(nbv[0]+2, nbv[1]+2, 'NBV', color='red', fontsize=8, weight='bold')
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    if title:
        ax.set_title(title, fontsize=10)
    return im

# ---------- Run ----------
os.makedirs(args.save_dir, exist_ok=True)
gc = GreedyCoverage(grid_size, fov_radius)
(viewpoints, coverage_ts, snapshots, nbvs,
 redundancy_ts, affinity_ts, time_ts) = gc.run(max_viewpoints, 1.0)

# ---------- Snapshots / animation ----------
if args.animate and len(snapshots) >= 2:
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    plot_frame(ax[0], viewpoints[:1], 1-snapshots[0].reshape(grid_size, grid_size), title="Initial")
    plot_frame(ax[1], viewpoints,    1-snapshots[-1].reshape(grid_size, grid_size), title="Final")
    fig.subplots_adjust(right=0.86)
    cax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(ax[1].images[0], cax=cax).set_label("Potential")
    snap = os.path.join(args.save_dir, f"{args.strategy}_seed{args.seed}_snapshots.png")
    plt.savefig(snap, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[Saved] Snapshots ➜ {snap}")

if args.animate:
    fig, ax = plt.subplots(figsize=(6,6)); fig.patch.set_facecolor('black')
    ani = animation.FuncAnimation(fig,
        lambda i: [plot_frame(ax, viewpoints[:i+1],
                              1-snapshots[i].reshape(grid_size, grid_size),
                              nbvs[i])],
        frames=len(snapshots), interval=100, blit=False)
    mp4 = os.path.join(args.save_dir, f"{args.strategy}_seed{args.seed}_anim.mp4")
    ani.save(mp4, writer='ffmpeg', fps=10, dpi=300); plt.close()
    print(f"[Saved] Animation  ➜ {mp4}")

# ---------- Final-state metrics ----------
N = len(viewpoints)
S = 50000
pts = np.random.rand(S,2) * grid_size
dist = np.linalg.norm(gc.wrap_distance(pts[:,None,:] - viewpoints[None,:,:]), axis=-1)
vis  = dist <= fov_radius                                     # (S, N)

# 1) visibility mask
viewpoint_point_assignments = vis.astype(np.bool_)

# 2) per-viewpoint contribution
viewpoint_contribution_hist = vis.sum(axis=0).astype(np.int32)

# 3) point redundancy histogram
redundancy_counts = vis.sum(axis=1)
point_redundancy_hist = np.bincount(redundancy_counts,
                                    minlength=redundancy_counts.max()+1).astype(np.int32)

# 4) viewpoint overlap (Jaccard)
overlap = np.zeros((N,N), dtype=np.float32)
for i in range(N):
    Vi = vis[:,i]
    for j in range(i, N):
        Vj  = vis[:,j]
        inter = np.logical_and(Vi,Vj).sum()
        union = np.logical_or (Vi,Vj).sum()
        overlap[i,j] = overlap[j,i] = inter / (union + epsilon)
np.fill_diagonal(overlap, 1.0)

# 5) functional isolation (<5 % mean overlap)
avg_ov = (overlap.sum(axis=1) - 1) / max(N-1,1)
functional_isolation_flags = (avg_ov < 0.05)

# ---------- Save ----------
npz = os.path.join(args.save_dir, f"{args.strategy}_seed{args.seed}_metrics.npz")
np.savez_compressed(npz,
    coverage       = coverage_ts,
    redundancy     = redundancy_ts,
    affinity       = affinity_ts,
    time           = time_ts,
    num_viewpoints = N,
    num_iterations = len(coverage_ts),
    # NEW analysis artefacts
    viewpoint_point_assignments = viewpoint_point_assignments,
    viewpoint_contribution_hist = viewpoint_contribution_hist,
    point_redundancy_hist       = point_redundancy_hist,
    viewpoint_overlap_matrix    = overlap,
    functional_isolation_flags  = functional_isolation_flags,
    final_viewpoints            = viewpoints      # <-- Add this line
)
print(f"[Saved] Metrics NPZ ➜ {npz}")
print("[Done] Greedy simulation complete.")
