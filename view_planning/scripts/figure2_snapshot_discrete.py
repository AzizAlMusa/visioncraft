#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import os

# ---------- Configuration ----------
FIG_WIDTH, FIG_HEIGHT = 7.2, 6.4
DPI = 600
GRID_SIZE = 100
RESOLUTION = 400
FOV_RADIUS = 20
NPZ_DIR = "./results2"

methods = [
    { "file": "greedy_seed0_metrics.npz", "title": "Greedy" },
    { "file": "rkga_seed0_metrics.npz", "title": "RKGA" },
    { "file": "sa_seed0_metrics.npz", "title": "Simulated Annealing" },
    { "file": "nbv_log_kattr10.00_krep0.25_seed0_metrics.npz", "title": "Potential Field" }
]

# ---------- Utilities ----------
def wrap_distance(diff):
    half = GRID_SIZE / 2
    return np.where(np.abs(diff) > half, -np.sign(diff) * (GRID_SIZE - np.abs(diff)), diff)

def compute_binary_visibility(viewpoints, resolution):
    x = np.linspace(0, GRID_SIZE, resolution)
    y = np.linspace(0, GRID_SIZE, resolution)
    X, Y = np.meshgrid(x, y)
    field_points = np.stack([X.ravel(), Y.ravel()], axis=-1)
    diff = field_points[:, None, :] - viewpoints[None, :, :]
    diff = wrap_distance(diff)
    dist = np.linalg.norm(diff, axis=-1)
    binary_visible = (dist <= FOV_RADIUS).any(axis=1).astype(np.float32)
    return binary_visible.reshape(resolution, resolution)

def plot_binary_field(ax, viewpoints, vis_mask, title):
    cmap = plt.cm.get_cmap("RdYlGn", 2)
    ax.imshow(vis_mask, cmap=cmap, origin='lower',
              extent=[0, GRID_SIZE, 0, GRID_SIZE],
              vmin=0.0, vmax=1.0, interpolation='nearest')

    for vp in viewpoints:
        for dx in (-GRID_SIZE, 0, GRID_SIZE):
            for dy in (-GRID_SIZE, 0, GRID_SIZE):
                circ = patches.Circle((vp[0]+dx, vp[1]+dy), FOV_RADIUS,
                                      edgecolor='white', facecolor='none', lw=0.5, alpha=0.6)
                ax.add_patch(circ)

    ax.scatter(viewpoints[:, 0], viewpoints[:, 1], c='white', s=4,
               edgecolors='black', linewidths=0.3)
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9, pad=4)
    ax.text(5, 5, f"{len(viewpoints)} viewpoints", fontsize=5,
            color='white', bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))

# ---------- Main ----------
plt.close('all')
fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

# Axes positions (tightened center gap)
axs = [
    fig.add_axes([0.11, 0.54, 0.38, 0.38], label='ax0'),  # top-left
    fig.add_axes([0.51, 0.54, 0.38, 0.38], label='ax1'),  # top-right
    fig.add_axes([0.11, 0.10, 0.38, 0.38], label='ax2'),  # bottom-left
    fig.add_axes([0.51, 0.10, 0.38, 0.38], label='ax3')   # bottom-right
]

# Plot
for ax, method in zip(axs, methods):
    data = np.load(os.path.join(NPZ_DIR, method["file"]))
    viewpoints = data["final_viewpoints"]
    vis_mask = compute_binary_visibility(viewpoints, RESOLUTION)
    plot_binary_field(ax, viewpoints, vis_mask, method["title"])

# Custom colormap for matching colors
cmap = plt.cm.get_cmap("RdYlGn", 2)
red_color = cmap(0)    # Not visible
green_color = cmap(1)  # Visible

# Bottom-left aligned title (small)
fig.text(0.13, 0.06, "Comparison of viewpoint set of different methods",
         ha='left', va='bottom', fontsize=8)

# Bottom-right aligned legend (aligned with right column)
legend_ax = fig.add_axes([0.51, 0.05, 0.38, 0.04], label='legend')
legend_ax.axis('off')
red_patch = mlines.Line2D([], [], color=red_color, marker='s', linestyle='None', markersize=8, label='Not visible')
green_patch = mlines.Line2D([], [], color=green_color, marker='s', linestyle='None', markersize=8, label='Visible')
legend_ax.legend(handles=[green_patch, red_patch], loc='center right', ncol=2, fontsize=7, frameon=False)

# Save
fig_path = os.path.join(NPZ_DIR, "figure2_final_binary_snapshots.png")
# plt.savefig(fig_path, dpi=DPI)
plt.savefig(fig_path, dpi=DPI, bbox_inches='tight', pad_inches=0.01)

print(f"[Saved] Final PNG ➜ {fig_path}")
