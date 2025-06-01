import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from collections import defaultdict
from scipy.integrate import simps

# === Configuration ===
results_dir = "./results"
expected_runs = 30
max_iter = 200
cutoff_iter = 80
grid_size = 80
fov_radius = 20
num_particles = 8

# === Load all files grouped by potential type ===
files = sorted(glob(os.path.join(results_dir, "*_seed*.npz")))
grouped_files = defaultdict(list)
for f in files:
    basename = os.path.basename(f)
    potential_type = basename.split("_seed")[0]
    grouped_files[potential_type].append(f)

# === Plateau Detection ===
def detect_plateau(signal, window_size=10, stddev_jump_factor=3.0, min_plateau_length=10):
    signal = np.array(signal)
    L = len(signal)
    tail_std = np.std(signal[-window_size:])
    for i in range(L - window_size * 2, 0, -1):
        curr_std = np.std(signal[i:i + window_size])
        if curr_std > stddev_jump_factor * tail_std:
            plateau_start = i + window_size
            if L - plateau_start >= min_plateau_length:
                return plateau_start
            break
    return -1

# === Plot Setup ===
x = np.arange(cutoff_iter)
fig, ax = plt.subplots(figsize=(7.16, 4.5))  # IEEE double-column width

# === Styling Parameters ===
label_fontsize = 12
tick_fontsize = 11
legend_fontsize = 11
title_fontsize = 14

# === Define Order and Colors ===
ordered_types = ["linear", "quadratic", "log", "inverse", "gaussian"]
colors = {
    "linear": "#DC143C",
    "quadratic": "#FF8C00", 
    "log": "#8A2BE2",
    "inverse": "#4169E1",
    "gaussian": "#2E8B57"
}

# === Plot Data ===
for ptype in ordered_types:
    if ptype not in grouped_files:
        continue
    flist = grouped_files[ptype]

    coverage_runs, redundancy_runs, affinity_runs = [], [], []

    for f in flist:
        data = np.load(f)
        c = data["coverage_per_iter"]
        r = data["redundancy_per_iter"]
        a = data["overlap_affinity_per_iter"]

        for arr in [c, r, a]:
            if len(arr) < max_iter:
                arr = np.pad(arr, (0, max_iter - len(arr)), mode="edge")

        coverage_runs.append(c[:cutoff_iter])
        redundancy_runs.append(r[:cutoff_iter])
        affinity_runs.append(a[:cutoff_iter])

    coverage_runs = np.array(coverage_runs)
    redundancy_runs = np.array(redundancy_runs)

    cov_mean, cov_std = np.mean(coverage_runs, axis=0), np.std(coverage_runs, axis=0)
    red_mean, red_std = np.mean(redundancy_runs, axis=0), np.std(redundancy_runs, axis=0)

    color = colors[ptype]

    ax.plot(x, cov_mean, color=color, linestyle='-', linewidth=1.0)
    ax.fill_between(x, cov_mean - cov_std, cov_mean + cov_std, color=color, alpha=0.2)

    ax.plot(x, red_mean, color=color, linestyle='--', linewidth=1.0, alpha=0.9)
    ax.fill_between(x, red_mean - red_std, red_mean + red_std, color=color, alpha=0.08)

    # Print metrics
    cov_auc = simps(cov_mean, dx=1)
    red_auc = simps(red_mean, dx=1)
    plateau_iter = detect_plateau(cov_mean)

    print(f"\n=== Summary Statistics ({ptype}) ===")
    print(f"Coverage AUC (0-{cutoff_iter}):   {cov_auc:.4f}")
    print(f"Redundancy AUC (0-{cutoff_iter}): {red_auc:.4f}")
    print(f"Final Coverage:                   {cov_mean[-1]:.4f}")
    print(f"Final Redundancy:                 {red_mean[-1]:.4f}")
    print(f"Coverage Plateau Iteration:       {plateau_iter}")

# === Axis and Title Formatting ===
ax.set_xlabel("Iteration", fontsize=label_fontsize)
ax.set_ylabel("Coverage / Redundancy", fontsize=label_fontsize)
ax.set_ylim(0, 1.05)
ax.set_title("Coverage and Redundancy by Potential Type", fontsize=title_fontsize, pad=10)
ax.tick_params(axis='both', labelsize=tick_fontsize)
ax.grid(True, alpha=0.3)

# === Manual Legend (Two Rows) ===
x_start = 0.0
x_spacing = 0.2
x_positions = [x_start + i * x_spacing for i in range(len(ordered_types))]
y_cov = -0.30
y_red = -0.42

for i, ptype in enumerate(ordered_types):
    color = colors[ptype]
    x_pos = x_positions[i]

        
    if i == 2:
        # Skip the "inverse" type for legend entries
                # Coverage line (solid)
        x_pos += 0.04
        ax.plot([x_pos - 0.015 , x_pos + 0.025], [y_cov]*2, transform=ax.transAxes,
                color=color, lw=1.5, solid_capstyle='butt', clip_on=False)
        ax.text(x_pos + 0.03, y_cov, f'{ptype} (C)', transform=ax.transAxes,
                fontsize=legend_fontsize, ha='left', va='center')

        # Redundancy line (dashed)
        ax.plot([x_pos - 0.015 , x_pos + 0.025 ], [y_red]*2, transform=ax.transAxes,
                color=color, lw=1.5, linestyle='--', dashes=(5, 3), clip_on=False)
        ax.text(x_pos + 0.03 , y_red, f'{ptype} (R)', transform=ax.transAxes,
                fontsize=legend_fontsize, ha='left', va='center')

    else:
        # Coverage line (solid)
        ax.plot([x_pos - 0.015, x_pos + 0.025], [y_cov]*2, transform=ax.transAxes,
                color=color, lw=1.5, solid_capstyle='butt', clip_on=False)
        ax.text(x_pos + 0.03, y_cov, f'{ptype} (C)', transform=ax.transAxes,
                fontsize=legend_fontsize, ha='left', va='center')

        # Redundancy line (dashed)
        ax.plot([x_pos - 0.015, x_pos + 0.025], [y_red]*2, transform=ax.transAxes,
                color=color, lw=1.5, linestyle='--', dashes=(5, 3), clip_on=False)
        ax.text(x_pos + 0.03, y_red, f'{ptype} (R)', transform=ax.transAxes,
                fontsize=legend_fontsize, ha='left', va='center')




# Explanation for (C) and (R)
ax.text(0.5, -0.54, '(C) = Coverage     (R) = Redundancy', transform=ax.transAxes,
        fontsize=legend_fontsize, ha='center', va='center', style='italic')

plt.tight_layout()
plt.subplots_adjust(bottom=0.43)
plt.savefig("potential_field_study.png", dpi=600, bbox_inches='tight')
plt.show()
