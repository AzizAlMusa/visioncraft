#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# ---------- Parameters ----------
grid_size = 100          # Field is [0, grid_size]
vis_resolution = 400     # High-res logic + visualization
fov_radius = 20
npz_dir = "./results2"
beta = 20.0
epsilon = 1e-6

methods = [
    { "file": "greedy_seed0_metrics.npz", "title": "Greedy" },
    { "file": "rkga_seed0_metrics.npz", "title": "RKGA" },
    { "file": "sa_seed0_metrics.npz", "title": "Simulated Annealing" },
    { "file": "nbv_log_kattr10.00_krep0.25_seed0_metrics.npz", "title": "Potential Field (Ours)" }
]

# ---------- Utilities ----------
def wrap_distance(diff):
    half = grid_size / 2
    return np.where(np.abs(diff) > half, -np.sign(diff) * (grid_size - np.abs(diff)), diff)

def compute_highres_field(viewpoints, resolution):
    x = np.linspace(0, grid_size, resolution)
    y = np.linspace(0, grid_size, resolution)
    X, Y = np.meshgrid(x, y)
    field_points = np.stack([X.ravel(), Y.ravel()], axis=-1)

    diff = field_points[:, None, :] - viewpoints[None, :, :]
    diff = wrap_distance(diff)
    dist = np.linalg.norm(diff, axis=-1) + epsilon
    vis = 1.0 / (1.0 + np.exp(beta * (dist / fov_radius - 1.0)))

    coverage = np.clip(vis.sum(axis=1), 0.0, 1.0)
    return coverage.reshape(resolution, resolution)

def plot_field(ax, viewpoints, coverage, title):
    im = ax.imshow(coverage, cmap='RdYlGn', origin='lower',
                   extent=[0, grid_size, 0, grid_size],
                   vmin=0.0, vmax=1.0, interpolation='bilinear')

    for vp in viewpoints:
        for dx in (-grid_size, 0, grid_size):
            for dy in (-grid_size, 0, grid_size):
                circ = patches.Circle((vp[0]+dx, vp[1]+dy), fov_radius,
                                      edgecolor='white', facecolor='none', lw=0.5, alpha=0.6)
                ax.add_patch(circ)

    ax.scatter(viewpoints[:, 0], viewpoints[:, 1], c='white', s=4,
               edgecolors='black', linewidths=0.3)

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9, pad=8)

    ax.text(5, 5, f"{len(viewpoints)} viewpoints", fontsize=5,
            color='white', bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))
    return im

# ---------- Main ----------
fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.0), dpi=600)
axs = axs.ravel()

for ax, method in zip(axs, methods):
    data = np.load(os.path.join(npz_dir, method["file"]))
    viewpoints = data["final_viewpoints"]
    coverage_map = compute_highres_field(viewpoints, vis_resolution)
    im = plot_field(ax, viewpoints, coverage_map, method["title"])

fig.subplots_adjust(left=0.06, right=0.82, top=0.95, bottom=0.05, wspace=0.12, hspace=0.12)

# Add colorbar
cbar_ax = fig.add_axes([0.86, 0.3, 0.02, 0.4])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Visibility (0 = not visible, 1 = fully visible)", fontsize=8)
cbar.ax.tick_params(labelsize=7)

# Save
fig_path = os.path.join(npz_dir, "figure2_final_snapshots.png")
plt.savefig(fig_path, dpi=600)
print(f"[Saved] Final PNG ➜ {fig_path}")
