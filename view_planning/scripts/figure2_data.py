#!/usr/bin/env python3
"""
Adaptive NBV / Potential-Field simulation
———————————————
• **All command-line args, filenames, snapshots, and animation remain unchanged.**
• Removes the interim / post-processing scalars you asked to drop:
      marginal_gain, stray_gap_views, avg_overlap_degree,
      num_isolated_viewpoints, uniformity_score
• Keeps the time-series arrays already used elsewhere (coverage, redundancy,
  affinity, time) so nothing downstream breaks.
• Re-implements the *final-configuration* data so that every quantity is
  derived **only once the viewpoints have finished adapting**:

      - viewpoint_point_assignments   ⟹ Boolean visibility mask
                                         shape = (num_sample_points, N_viewpoints)
      - viewpoint_contribution_hist   ⟹ #points seen by each viewpoint
      - point_redundancy_hist         ⟹ histogram: how many VPs see each point
      - viewpoint_overlap_matrix      ⟹ Jaccard overlap (|∩| / |∪|)  N×N matrix
      - functional_isolation_flags    ⟹ True if a VP’s *average* overlap < 5 %

Everything else (logic, animation, Adam-type motion, smart insertion, etc.)
is exactly as before.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse, os, time

# ---------- Arguments ----------
parser = argparse.ArgumentParser()
parser.add_argument("--strategy", type=str, default="nbv", choices=["nbv", "random"])
parser.add_argument("--potential_type", type=str, default="log")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save_dir", type=str, default="./results2")
parser.add_argument("--animate", action="store_true")
parser.add_argument("--smart_nbv_insertion", action="store_true")
parser.add_argument("--k_attr", type=float, default=10.0)
parser.add_argument("--k_rep", type=float, default=0.25)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

# ---------- Globals ----------
np.random.seed(args.seed)
grid_size, fov_radius = 100, 20
epsilon               = 1e-6
frames_per_stage       = 10
max_viewpoints         = 100
T_motion_rough, T_motion_fine = 1.5, 0.15
window_size            = 5

x, y          = np.arange(grid_size), np.arange(grid_size)
X, Y          = np.meshgrid(x, y)
field_points  = np.stack([X.ravel(), Y.ravel()], axis=-1)

# ---------- Field ----------
class Field:
    def __init__(self, grid_size, fov_radius, use_wrapping=True):
        self.grid_size, self.fov_radius = grid_size, fov_radius
        self.use_wrapping = use_wrapping
        self.visibility   = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.potential    = np.zeros((grid_size, grid_size))
        self.field_points = field_points          # (N², 2)

    def wrap_distance(self, diff):
        if not self.use_wrapping:
            return diff
        half = self.grid_size / 2
        return np.where(np.abs(diff) > half,
                        -np.sign(diff) * (self.grid_size - np.abs(diff)), diff)

    def update_visibility(self, particles, beta=20.0):
        self.visibility.fill(0.0)
        diff   = self.field_points[:, None, :] - particles[None, :, :]
        diff   = self.wrap_distance(diff)
        dist   = np.linalg.norm(diff, axis=-1) + epsilon
        s      = 1.0 / (1.0 + np.exp(beta * (dist / self.fov_radius - 1.0)))
        self.visibility.ravel()[:] = np.clip(s.sum(axis=1), 0.0, 1.0)

    def compute_potential(self, particles, alpha=1.0):
        diff   = self.field_points[:, None, :] - particles[None, :, :]
        diff   = self.wrap_distance(diff)
        dist   = np.linalg.norm(diff, axis=-1) + epsilon
        pot    = alpha * (1.0 - self.visibility.ravel()) * np.log(dist).sum(axis=1)
        self.potential = pot.reshape(self.grid_size, self.grid_size)

    def compute_force(self, particles, k_attr=0.4, k_rep=1.0, alpha=1.0):
        # Attractive
        diff   = self.field_points[:, None, :] - particles[None, :, :]
        diff   = self.wrap_distance(diff)
        dist   = np.linalg.norm(diff, axis=-1) + epsilon
        dirn   = diff / dist[..., None]
        need   = alpha * (1.0 - self.visibility.ravel())[:, None] / dist
        need[dist < epsilon] = 0.0
        attractive = (need[..., None] * dirn)\
                     .reshape(self.grid_size, self.grid_size, particles.shape[0], 2).sum(axis=(0,1))
        # Repulsive (RBF)
        pdiff  = particles[:, None, :] - particles[None, :, :]
        pdiff  = self.wrap_distance(pdiff)
        pdist  = np.linalg.norm(pdiff, axis=-1) + epsilon
        sigma, amp = 10.0, 100.0
        rep    = -(amp * pdiff * (-pdist[..., None] / sigma**2) *
                   np.exp(-(pdist**2) / (2*sigma**2))[..., None]).sum(axis=1)
        return k_attr * attractive + k_rep * rep

    def monte_carlo_coverage(self, particles, S=10000, thresh=0.25):
        pts  = np.random.rand(S,2) * self.grid_size
        d    = np.linalg.norm(self.wrap_distance(pts[:,None,:]-particles[None,:,:]), axis=-1)
        vis  = d <= self.fov_radius
        return np.mean(vis.sum(axis=1) > thresh)

# ---------- Helper: frame plot ----------
def plot_frame(ax, pts, pot, nbv=None, title=None):
    ax.clear()
    im = ax.imshow(pot, cmap='viridis', origin='lower',
                   extent=[0,grid_size,0,grid_size],
                   vmin=pot.min(), vmax=pot.max())
    for p in pts:
        for dx in (-grid_size,0,grid_size):
            for dy in (-grid_size,0,grid_size):
                ax.add_patch(plt.Circle((p[0]+dx, p[1]+dy), fov_radius,
                              edgecolor='white', facecolor='none', alpha=0.2, lw=1))
    ax.scatter(pts[:,0], pts[:,1], c='deepskyblue', s=30,
               edgecolors='white', linewidths=0.6)
    if nbv is not None and args.strategy == "greedy":
        ax.scatter(*nbv, s=60, c='red', edgecolors='white', lw=1.5)
        ax.text(nbv[0]+2, nbv[1]+2, 'NBV', color='red', fontsize=8, weight='bold')
    ax.set_xlim(0,grid_size); ax.set_ylim(0,grid_size)
    if title: ax.set_title(title, fontsize=10)
    return im

# ---------- Setup ----------
os.makedirs(args.save_dir, exist_ok=True)
field        = Field(grid_size, fov_radius)
particles    = np.random.rand(1,2) * grid_size
m = np.zeros_like(particles); v = np.zeros_like(particles)
t_adam       = 0
coverage_ts, redundancy_ts, affinity_ts, time_ts = [], [], [], []
frames_pts, frames_pot, frames_nbv = [], [], []
recent_moves = []
start_t      = time.time()

# ---------- Main loop ----------
while True:
    field.update_visibility(particles)
    field.compute_potential(particles)
    coverage = field.monte_carlo_coverage(particles)
    coverage_ts.append(coverage)
    time_ts.append(time.time() - start_t)

    nbv = (field.field_points[np.argmax(field.potential)]
           if args.strategy=="nbv" else np.random.rand(2)*grid_size)

    frames_pts.append(particles.copy())
    frames_pot.append(field.potential.copy())
    frames_nbv.append(nbv.copy())

    # Adam-like motion
    forces  = field.compute_force(particles, k_attr=args.k_attr, k_rep=args.k_rep)
    t_adam += 1
    m = 0.9*m + 0.1*forces
    v = 0.999*v + 0.001*(forces**2)
    step = 5 * (m/(1-0.9**t_adam)) / (np.sqrt(v/(1-0.999**t_adam)) + epsilon)
    particles = (particles + step) % grid_size

    move_mag = np.linalg.norm(step, axis=1).mean()
    recent_moves.append(move_mag)
    if len(recent_moves) > window_size: recent_moves.pop(0)

    # Redundancy / affinity for this frame
    smpl = np.random.rand(10000,2)*grid_size
    d    = np.linalg.norm(field.wrap_distance(smpl[:,None,:] - particles[None,:,:]), axis=-1)
    vis  = d <= fov_radius
    cnt  = vis.sum(axis=1)
    redundancy_ts.append(np.mean(cnt>1))
    affinity_ts.append(np.mean(cnt[cnt>0]) if np.any(cnt>0) else 0.0)

    # Insertion logic
    if args.smart_nbv_insertion:
        if len(particles) <= 2:
            insert = (len(frames_pts)-1) % frames_per_stage == 0
        else:
            T = T_motion_fine if coverage >= .95 else T_motion_rough
            insert = len(recent_moves)==window_size and all(mv < T for mv in recent_moves)
    else:
        insert = (len(frames_pts)-1) % frames_per_stage == 0

    if args.verbose:
        print(f"[{len(frames_pts)-1:03d}] move={move_mag:5.3f} cov={coverage:6.3f} "
              f"vp={len(particles):3d} add={'✔' if insert else '—'}")

    if insert:
        particles = np.vstack([particles, nbv[None,:]])
        m = np.zeros_like(particles); v = np.zeros_like(particles)
        t_adam = 0; recent_moves.clear()

    if coverage >= 1.0 or len(particles) >= max_viewpoints: break

# ---------- Animation / snapshots ----------
if args.animate:
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    plot_frame(ax[0], frames_pts[0], frames_pot[0], title="Initial")
    plot_frame(ax[1], particles, field.potential, title="Final")
    fig.subplots_adjust(right=0.86)
    cax = fig.add_axes([0.88,0.15,0.02,0.7])
    fig.colorbar(ax[1].images[0], cax=cax).set_label("Potential")
    snap = os.path.join(args.save_dir,
           f"{args.strategy}_{args.potential_type}_seed{args.seed}_snapshots.png")
    plt.savefig(snap,dpi=300,bbox_inches='tight'); plt.close()
    print(f"[Saved] Snapshots ➜ {snap}")

    fig, ax = plt.subplots(figsize=(6,6)); fig.patch.set_facecolor('black')
    ani = animation.FuncAnimation(fig,
          lambda i: [plot_frame(ax, frames_pts[i], frames_pot[i], frames_nbv[i])],
          frames=len(frames_pts), interval=100, blit=False)
    mp4 = os.path.join(args.save_dir,
          f"{args.strategy}_{args.potential_type}_seed{args.seed}_anim.mp4")
    ani.save(mp4, writer='ffmpeg', fps=10, dpi=300); plt.close()
    print(f"[Saved] Animation  ➜ {mp4}")

# ---------- Final-state analysis ----------
N          = len(particles)
S_final    = 50000
pts_final  = np.random.rand(S_final,2) * grid_size
dist_final = np.linalg.norm(
                field.wrap_distance(pts_final[:,None,:] - particles[None,:,:]), axis=-1)
vis_mask   = dist_final <= fov_radius                     # (S_final, N)

# 1) viewpoint_point_assignments  (visibility mask)
viewpoint_point_assignments = vis_mask.astype(np.bool_)

# 2) viewpoint_contribution_hist (#points each VP sees)
viewpoint_contribution_hist = vis_mask.sum(axis=0).astype(np.int32)

# 3) point_redundancy_hist (#VPs per point)
redundancy_counts = vis_mask.sum(axis=1)
point_redundancy_hist = np.bincount(redundancy_counts,
                                    minlength=redundancy_counts.max()+1).astype(np.int32)

# 4) viewpoint_overlap_matrix (Jaccard)
overlap = np.zeros((N,N), dtype=np.float32)
for i in range(N):
    Vi = vis_mask[:,i]
    for j in range(i, N):
        Vj = vis_mask[:,j]
        inter = np.logical_and(Vi, Vj).sum()
        union = np.logical_or (Vi, Vj).sum()
        overlap[i,j] = overlap[j,i] = inter / (union + epsilon)
np.fill_diagonal(overlap, 1.0)

# 5) functional_isolation_flags (avg overlap < 5 %)
avg_ov = (overlap.sum(axis=1) - 1) / np.maximum(N-1,1)
functional_isolation_flags = (avg_ov < 0.05)

# ---------- Save ----------
npz = os.path.join(args.save_dir,
       f"{args.strategy}_{args.potential_type}_kattr{args.k_attr:.2f}_krep{args.k_rep:.2f}_seed{args.seed}_metrics.npz")
np.savez_compressed(npz,
    coverage=np.array(coverage_ts),
    redundancy=np.array(redundancy_ts),
    affinity=np.array(affinity_ts),
    time=np.array(time_ts),
    num_viewpoints=N,
    num_iterations=len(frames_pts),
    # NEW data
    viewpoint_point_assignments=viewpoint_point_assignments,
    viewpoint_contribution_hist=viewpoint_contribution_hist,
    point_redundancy_hist=point_redundancy_hist,
    viewpoint_overlap_matrix=overlap,
    functional_isolation_flags=functional_isolation_flags,
    final_viewpoints=particles              # <-- ADD THIS LINE

)
print(f"[Saved] Metrics NPZ ➜ {npz}")
print("[Done] Simulation complete.")
