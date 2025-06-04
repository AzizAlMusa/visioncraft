#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.stats import entropy

def compute_metrics(assignments):
    num_points, num_viewpoints = assignments.shape

    # Point redundancy → how many viewpoints see each point
    seen_counts = assignments.sum(axis=1)
    redundancy_hist = np.bincount(seen_counts, minlength=10)
    redundancy_percent = redundancy_hist / redundancy_hist.sum() * 100

    # Exclusive area per viewpoint (i.e. points only seen by 1 VP)
    exclusive_mask = (seen_counts == 1)
    exclusive_contributions = np.zeros(num_viewpoints, dtype=float)
    for idx in np.where(exclusive_mask)[0]:
        vp = assignments[idx].argmax()
        exclusive_contributions[vp] += 1
    exclusive_contributions = (exclusive_contributions / num_points) * 100  # → % of area

    # Overlap matrix + entropy
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

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--npz_pf", type=str, required=True)
parser.add_argument("--npz_greedy", type=str, required=True)
parser.add_argument("--npz_rkga", type=str, required=True)
parser.add_argument("--npz_sa", type=str, required=True)
parser.add_argument("--save_path", type=str, default=None)
args = parser.parse_args()

# ---------- Load and compute ----------
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

# ---------- Colors and Labels ----------
methods = [
    ("PF", metrics_pf, "#33a02c"),
    ("Greedy", metrics_gr, "#1f78b4"),
    ("RKGA", metrics_rkga, "#ff7f00"),
    ("SA", metrics_sa, "#6a3d9a"),
]

# ---------- Plotting ----------
fig, axs = plt.subplots(1, 4, figsize=(19.2, 3.8))
fig.suptitle("Comparison of View Planning Methods", fontsize=13)

# 1. Redundancy %
x = np.arange(6)
bar_width = 0.18
for i, (label, m, color) in enumerate(methods):
    if m:
        axs[0].bar(x + i*bar_width - bar_width*1.5, m["redundancy_percent"], width=bar_width, label=label, color=color)
axs[0].set_title("Point Redundancy (% of Area)", fontsize=10)
axs[0].set_xlabel("# Viewpoints per Point", fontsize=9)
axs[0].set_ylabel("% of Area", fontsize=9)
axs[0].legend(frameon=False, fontsize=8)
axs[0].tick_params(labelsize=8)

# 2. Exclusive Area Boxplot
data = [m["exclusive"] for _, m, _ in methods if m]
labels = [label for label, m, _ in methods if m]
colors = [color for _, m, color in methods if m]
axs[1].boxplot(data, labels=labels, patch_artist=True,
               boxprops=dict(facecolor='lightgray'), medianprops=dict(color='black'))
for i, ex in enumerate(data):
    axs[1].scatter([i+1]*len(ex), ex, s=12, alpha=0.6, color=colors[i])
axs[1].set_title("Exclusive Area per Viewpoint", fontsize=10)
axs[1].set_ylabel("% of Area", fontsize=9)
axs[1].tick_params(labelsize=8)

# 3. Overlap Entropy Boxplot
data = [m["entropy"] for _, m, _ in methods if m]
axs[2].boxplot(data, labels=labels, patch_artist=True,
               boxprops=dict(facecolor='lightgray'), medianprops=dict(color='black'))
for i, en in enumerate(data):
    axs[2].scatter([i+1]*len(en), en, s=12, alpha=0.6, color=colors[i])
axs[2].set_title("Overlap Entropy", fontsize=10)
axs[2].set_ylabel("Entropy", fontsize=9)
axs[2].tick_params(labelsize=8)

# 4. Entropy vs Exclusive
for (label, m, color) in methods:
    if m:
        axs[3].scatter(m["entropy"], m["exclusive"], s=18, alpha=0.7, label=label, color=color)
axs[3].set_title("Entropy vs Exclusive Area", fontsize=10)
axs[3].set_xlabel("Overlap Entropy", fontsize=9)
axs[3].set_ylabel("Exclusive Area (%)", fontsize=9)
axs[3].legend(frameon=False, fontsize=8)
axs[3].tick_params(labelsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.92])
if args.save_path:
    plt.savefig(args.save_path, dpi=300, bbox_inches='tight')
    print(f"[Saved] {args.save_path}")
else:
    plt.show()
