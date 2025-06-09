#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.stats import entropy
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import os

def compute_metrics(assignments):
    num_points, num_viewpoints = assignments.shape
    seen_counts = assignments.sum(axis=1)
    redundancy_hist = np.bincount(seen_counts, minlength=10)
    redundancy_percent = redundancy_hist / redundancy_hist.sum() * 100

    exclusive_mask = (seen_counts == 1)
    exclusive_contributions = np.zeros(num_viewpoints, dtype=float)
    for idx in np.where(exclusive_mask)[0]:
        vp = assignments[idx].argmax()
        exclusive_contributions[vp] += 1
    exclusive_contributions = (exclusive_contributions / num_points) * 100

    overlap_matrix = np.zeros((num_viewpoints, num_viewpoints), dtype=np.float32)
    for i in range(num_viewpoints):
        vi = assignments[:, i]
        for j in range(i, num_viewpoints):
            vj = assignments[:, j]
            inter = np.logical_and(vi, vj).sum()
            union = np.logical_or(vi, vj).sum()
            overlap_matrix[i, j] = overlap_matrix[j, i] = inter / (union + 1e-6)
    np.fill_diagonal(overlap_matrix, 1.0)

    overlap_entropies = []
    for i in range(num_viewpoints):
        row = np.delete(overlap_matrix[i], i)
        norm_row = row / (row.sum() + 1e-6)
        overlap_entropies.append(entropy(norm_row))

    return {
        "redundancy_percent": redundancy_percent[:6],
        "exclusive": exclusive_contributions,
        "entropy": overlap_entropies,
    }

def draw_cov_ellipse(xs, ys, ax, n_std=1.0, facecolor='none', edgecolor='black', **kwargs):
    cov = np.cov(xs, ys)
    if cov.shape != (2, 2):
        return
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      edgecolor=edgecolor,
                      linestyle='--',
                      linewidth=1,
                      alpha=0.7,
                      **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x = np.mean(xs)
    mean_y = np.mean(ys)
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--npz_pf", type=str, required=True)
parser.add_argument("--npz_greedy", type=str, required=True)
parser.add_argument("--npz_rkga", type=str, required=True)
parser.add_argument("--npz_sa", type=str, required=True)
parser.add_argument("--save_path", type=str, default="./results2/overlap_redundancy_comparison.pdf")
args = parser.parse_args()

def load_and_compute(npz_path, label):
    try:
        data = np.load(npz_path)
        assign = data["viewpoint_point_assignments"]
        if assign.shape[0] < assign.shape[1]:
            assign = assign.T
        return compute_metrics(assign)
    except Exception as e:
        print(f"[Warning] Could not load {label}: {e}")
        return None

metrics_pf   = load_and_compute(args.npz_pf, "PF")
metrics_gr   = load_and_compute(args.npz_greedy, "Greedy")
metrics_rkga = load_and_compute(args.npz_rkga, "RKGA")
metrics_sa   = load_and_compute(args.npz_sa, "SA")

methods = [
    ("PF", metrics_pf, "#33a02c"),
    ("Greedy", metrics_gr, "#1f78b4"),
    ("RKGA", metrics_rkga, "#ff7f00"),
    ("SA", metrics_sa, "#6a3d9a"),
]

# ---------- Plotting ----------
fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.4), dpi=600)

# 1. Coverage Frequency Distribution
x = np.arange(6)
bar_width = 0.18
for i, (label, m, color) in enumerate(methods):
    if m:
        axs[0].bar(x + i*bar_width - bar_width*1.5, m["redundancy_percent"], width=bar_width, label=label, color=color)
axs[0].set_title("Coverage Frequency Distribution", fontsize=6)
axs[0].set_xlabel("Times a Region is Observed", fontsize=6)
axs[0].set_ylabel("% of Area", fontsize=6)
axs[0].legend(frameon=False, fontsize=7)
axs[0].tick_params(labelsize=7)

# 2. Entropy vs Exclusive Area with Ellipses
label_offsets = {
    "PF": (0.1, 0.00),
    "Greedy": (-0.6, 0.15),
    "RKGA": (0.1, -0.25),
    "SA": (0.1, 0.1),
}
axs[1].set_title("Overlap Entropy vs Viewpoint Unique Contribution", fontsize=6)
axs[1].set_xlabel("Unique Contribution / Viewpoint", fontsize=6)
axs[1].set_ylabel("Overlap Entropy", fontsize=6)
axs[1].tick_params(labelsize=7)

for label, m, color in methods:
    if m:
        xs = m["exclusive"]
        ys = m["entropy"]
        mean_x = np.mean(xs)
        mean_y = np.mean(ys)
        axs[1].scatter(xs, ys, s=5, alpha=0.25, color=color)
        axs[1].scatter(mean_x, mean_y, s=20, color=color, edgecolor='black', zorder=3)
        draw_cov_ellipse(xs, ys, axs[1], edgecolor=color)
        dx, dy = label_offsets[label]
        axs[1].annotate(label, (mean_x + dx, mean_y + dy), fontsize=6, color=color)

plt.tight_layout(rect=[0, 0, 1, 1.0])
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
plt.savefig(args.save_path, bbox_inches='tight')
print(f"[Saved] {args.save_path}")

print("\n--- Entropy vs Unique Contribution Data ---")
for label, m, _ in methods:
    if m:
        xs = m["exclusive"]
        ys = m["entropy"]
        print(f"\nMethod: {label}")
        print("Exclusive Area (%):", np.round(xs, 3).tolist())
        print("Overlap Entropy:", np.round(ys, 3).tolist())
