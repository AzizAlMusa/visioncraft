import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from collections import defaultdict

# === Configuration ===
results_dir = "./results"
max_iter    = 200
cutoff_iter = 80

files = sorted(glob(os.path.join(results_dir, "*_seed*.npz")))
grouped_files = defaultdict(list)
for f in files:
    potential_type = os.path.basename(f).split("_seed")[0]
    grouped_files[potential_type].append(f)

def pad_to(arr, L):
    arr = np.asarray(arr)
    return np.pad(arr, (0, max(0, L - len(arr))), mode="edge")

# === Plot setup (IEEE one-column width) ===
x = np.arange(cutoff_iter)
fig, (ax_cov, ax_red) = plt.subplots(
    2, 1, figsize=(3.5, 4.0), sharex=True, gridspec_kw={"hspace": 0.16}
)

fs = 10
ordered_types = ["linear", "quadratic", "log", "inverse", "gaussian"]
colors = {
    "linear":"#DC143C", "quadratic":"#FF8C00", "log":"#8A2BE2",
    "inverse":"#4169E1", "gaussian":"#2E8B57"
}

handles, labels = [], []

# === Plot data ===
for ptype in ordered_types:
    if ptype not in grouped_files:
        continue
    coverage_runs, redundancy_runs = [], []
    for f in grouped_files[ptype]:
        data = np.load(f)
        c = pad_to(data["coverage_per_iter"],   max_iter)[:cutoff_iter]
        r = pad_to(data["redundancy_per_iter"], max_iter)[:cutoff_iter]
        coverage_runs.append(c); redundancy_runs.append(r)

    cov_mean = np.mean(coverage_runs, axis=0)
    cov_std  = np.std(coverage_runs, axis=0)
    red_mean = np.mean(redundancy_runs, axis=0)
    red_std  = np.std(redundancy_runs, axis=0)
    color    = colors[ptype]

    h_cov, = ax_cov.plot(x, cov_mean, color=color, lw=1.0, label=ptype)
    ax_cov.fill_between(x, cov_mean - cov_std, cov_mean + cov_std, color=color, alpha=0.20)

    ax_red.plot(x, red_mean, color=color, lw=1.0)
    ax_red.fill_between(x, red_mean - red_std, red_mean + red_std, color=color, alpha=0.08)

    handles.append(h_cov); labels.append(ptype)

# === Axis formatting ===
ax_cov.set_ylabel("Coverage", fontsize=fs)
ax_cov.set_ylim(0, 1.05)
ax_cov.set_title("Coverage", fontsize=fs, pad=6)
ax_cov.tick_params(axis='both', labelsize=fs)
ax_cov.grid(True, alpha=0.3)

ax_red.set_xlabel("Iteration", fontsize=fs)
ax_red.set_ylabel("Redundancy", fontsize=fs)
ax_red.set_ylim(0, 1.05)
ax_red.set_title("Redundancy", fontsize=fs, pad=6)
ax_red.tick_params(axis='both', labelsize=fs)
ax_red.grid(True, alpha=0.3)

# === Legend inside, below plots (fully visible) ===
plt.subplots_adjust(bottom=0.26, top=0.95, left=0.18, right=0.97)
legend = fig.legend(
    handles, labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.06),   # slightly higher so it's fully in-bounds
    fontsize=fs,
    ncol=len(ordered_types),      # one line
    frameon=False,
    columnspacing=0.55,           # tighter spacing between entries
    handlelength=0.9,             # shorter line in legend
    handletextpad=0.25            # tighter text-to-line gap
)

# === Save with minimal borders ===
fig.savefig("potential_field_study2.png", dpi=600, bbox_inches=None, pad_inches=0.0)
plt.show()
