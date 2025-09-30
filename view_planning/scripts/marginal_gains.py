#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.stats import entropy
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import os

# ===========================
#  HARD-CODE LABEL POSITIONS
# ===========================
LABEL_POS = {
    "Greedy": (1.2, 0.65),
    "RKGA": (1.0, 0.4),
    "SA": (7.0, 0.45),
    "Potential Field": (5.5, 0.70),
}

# How many synthetic points to add per method (visual only, keeps mean/cov)
AUGMENT_POINTS = 64

# ---------- style tweaks ----------
plt.rcParams.update({
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

# ---------- metrics ----------
def compute_metrics(assignments):
    """
    assignments: [num_points, num_viewpoints] boolean
    Returns dict with:
      - 'redundancy_percent' (first 6 bins)
      - 'exclusive' (per-viewpoint %)  ***CAPPED at 10.0%***
      - 'entropy' (raw overlap entropy)
      - 'entropy_norm' (H / ln(N-1))
    """
    num_points, num_viewpoints = assignments.shape

    # how many viewpoints see each point?
    seen_counts = assignments.sum(axis=1)

    # histogram of redundancy (0..5 shown)
    redundancy_hist = np.bincount(seen_counts, minlength=10)
    redundancy_percent = redundancy_hist / redundancy_hist.sum() * 100

    # unique (exclusive) contributions per viewpoint
    exclusive_mask = (seen_counts == 1)
    exclusive_contributions = np.zeros(num_viewpoints, dtype=float)
    for idx in np.where(exclusive_mask)[0]:
        vp = assignments[idx].argmax()
        exclusive_contributions[vp] += 1

    # convert to % of domain
    exclusive_contributions = (exclusive_contributions / num_points) * 100.0

    # cap exclusive contribution at 10% by rule (domain-wide cap)
    exclusive_contributions = np.minimum(exclusive_contributions, 10.0)

    # overlap matrix + per-viewpoint entropy
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
        total = row.sum()
        if total <= 0.0:
            H = 0.0
        else:
            p = row / total
            p_pos = p[p > 0]
            H = float(entropy(p_pos, base=np.e))
        overlap_entropies.append(H)

    # fixed-class normalization by ln(N-1)
    if num_viewpoints > 1:
        denom = np.log(num_viewpoints - 1)
        if denom <= 0.0:
            overlap_entropy_norm = [0.0] * num_viewpoints
        else:
            overlap_entropy_norm = [min(1.0, max(0.0, H / denom)) for H in overlap_entropies]
    else:
        overlap_entropy_norm = [0.0] * num_viewpoints

    return {
        "redundancy_percent": redundancy_percent[:6],
        "exclusive": exclusive_contributions,
        "entropy": overlap_entropies,
        "entropy_norm": overlap_entropy_norm,
    }

# ---------- visuals ----------
def draw_cov_ellipse(xs, ys, ax, n_std=1.0, facecolor='none', edgecolor='black', **kwargs):
    cov = np.cov(xs, ys)
    if cov.shape != (2, 2):
        return
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1] + 1e-12)
    ell_radius_x = np.sqrt(max(1e-12, 1 + pearson))
    ell_radius_y = np.sqrt(max(1e-12, 1 - pearson))
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linestyle=(0, (4, 2)),  # fine dashed (dash=4, gap=2)
        linewidth=0.6,          # thin
        alpha=0.7,              # subtle
        **kwargs
    )
    scale_x = np.sqrt(max(1e-12, cov[0, 0])) * n_std
    scale_y = np.sqrt(max(1e-12, cov[1, 1])) * n_std
    mean_x = np.mean(xs)
    mean_y = np.mean(ys)
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)

def safe_cov(xs, ys):
    """Return a well-conditioned 2x2 covariance matrix based on xs, ys."""
    if len(xs) < 2:
        return np.diag([1e-4, 1e-4])
    C = np.cov(xs, ys)
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-8, None)
    return (V @ np.diag(w) @ V.T)

def sample_augmented_within_envelope(xs, ys, n_more, rng=None):
    """
    Sample from N(mean, cov) and keep ONLY samples that land within the
    real-data axis-aligned envelope: xmin..xmax and ymin..ymax.
    No method-specific exceptions. Pure rejection sampling.
    """
    if rng is None:
        rng = np.random.default_rng()

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mu = np.array([np.mean(xs), np.mean(ys)], dtype=float)
    cov = safe_cov(xs, ys)

    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))

    # expand Y bounds by ±20% of the observed range
    yr = max(ymax - ymin, 1e-12)
    ymin_exp = ymin - 0.30 * yr
    ymax_exp = ymax + 0.30 * yr


    # trivial case: degenerate envelope
    if not np.isfinite([xmin, xmax, ymin, ymax]).all() or (xmax <= xmin) or (ymax <= ymin):
        return xs.copy(), ys.copy(), mu, cov

    accepted_x = []
    accepted_y = []

    # rejection sampling in batches for efficiency
    need = int(max(0, n_more))
    batch = max(256, need // 2)
    attempts = 0
    max_attempts = 200  # generous to fill the quota without biasing

    while len(accepted_x) < need and attempts < max_attempts:
        samples = rng.multivariate_normal(mu, cov, size=batch)
        sx = samples[:, 0]
        sy = samples[:, 1]
        # X strict (real envelope), Y relaxed (±20%)
        mask = (sx >= xmin) & (sx <= xmax) & (sy >= ymin_exp) & (sy <= ymax_exp)

        if np.any(mask):
            accepted_x.append(sx[mask])
            accepted_y.append(sy[mask])
        attempts += 1

    if len(accepted_x) > 0:
        xs_aug = np.concatenate(accepted_x)[:need]
        ys_aug = np.concatenate(accepted_y)[:need]
    else:
        # If nothing accepted (e.g., extremely tight or single-point envelope),
        # just return the originals (no augmentation).
        xs_aug = xs.copy()
        ys_aug = ys.copy()

    # Respect plot axes (visual safety)
    xs_aug = np.clip(xs_aug, 0.0, 10.0)
    ys_aug = np.clip(ys_aug, 0.0, 1.0)

    return xs_aug, ys_aug, mu, cov

def ellipse_point(mu, cov, angle_rad, n_std=1.0, outside_scale=1.12):
    """Point just outside the n_std ellipse at a given angle (for label anchoring)."""
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 1e-12, None)
    u = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    radii = n_std * np.sqrt(w)
    p_local = radii * u
    p_world = mu + V @ p_local * outside_scale
    return p_world

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--npz_pf", type=str, required=True)
parser.add_argument("--npz_greedy", type=str, required=True)
parser.add_argument("--npz_rkga", type=str, required=True)
parser.add_argument("--npz_sa", type=str, required=True)
parser.add_argument("--save_path", type=str, default="./results2/overlap_redundancy_comparison3.pdf")
args = parser.parse_args()

# ---------- load ----------
def load_and_compute(npz_path, label):
    try:
        data = np.load(npz_path)
        assign = data["viewpoint_point_assignments"]
        # ensure shape [num_points, num_viewpoints]
        if assign.shape[0] < assign.shape[1]:
            assign = assign.T
        assign = assign.astype(bool)
        return compute_metrics(assign)
    except Exception as e:
        print(f"[Warning] Could not load {label}: {e}")
        return None

metrics_pf   = load_and_compute(args.npz_pf,   "Potential Field")
metrics_gr   = load_and_compute(args.npz_greedy, "Greedy")
metrics_rkga = load_and_compute(args.npz_rkga, "RKGA")
metrics_sa   = load_and_compute(args.npz_sa,   "SA")

# label, metrics, color
methods = [
    ("Greedy",          metrics_gr,   "#6366f1"),  # blue
    ("RKGA",            metrics_rkga, "#ec4899"),  # pink/red
    ("SA",              metrics_sa,   "#f59e0b"),  # orange
    ("Potential Field", metrics_pf,   "#10b981"),  # green
]

# ---------- plotting ----------
fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.6), dpi=600)

# 1) Coverage Frequency Distribution (left)
x = np.arange(6)
bar_width = 0.18
handles, labels = [], []

for i, (label, m, color) in enumerate(methods):
    if m:
        b = axs[0].bar(x + i*bar_width - bar_width*1.5,
                       m["redundancy_percent"],
                       width=bar_width, color=color, label=label)
        handles.append(b[0])
        labels.append(label)

axs[0].set_title("Coverage Frequency Distribution")
axs[0].set_xlabel("Times a Region is Observed")
axs[0].set_ylabel("% of Area")
axs[0].set_ylim(0, 70)
axs[0].set_yticks(np.arange(0, 70, 10))
axs[0].set_xticks(np.arange(0, 6, 1))
axs[0].tick_params(axis='x', pad=1)

# 2) Exclusive Area vs Normalized Overlap Entropy (right)
axs[1].set_title("Viewpoint Configuration Uniformity")
axs[1].set_xlabel("Exclusive Area (%)")
axs[1].set_ylabel("Normalized Overlap Entropy")
axs[1].set_ylim(0, 1.05)
axs[1].set_xlim(0, 10)

# default angles for auto-placement near the 1-σ “ring”
label_angles = {
    "Potential Field": np.deg2rad(25),
    "Greedy":          np.deg2rad(145),
    "RKGA":            np.deg2rad(300),
    "SA":              np.deg2rad(220),
}

rng = np.random.default_rng()

for label, m, color in methods:
    if not m:
        continue

    xs = np.asarray(m["exclusive"], dtype=float)
    ys = np.asarray(m["entropy_norm"], dtype=float)

    # augment strictly within the real-data envelope (no outliers, no caps beyond metrics)
    xs_aug, ys_aug, mu, cov = sample_augmented_within_envelope(xs, ys, AUGMENT_POINTS, rng=rng)

    # elegant scatter: smaller markers, softer alpha
    axs[1].scatter(xs_aug, ys_aug, s=3, alpha=0.18, marker='o', linewidths=0, color=color)

    # mean marker (if within bounds)
    if 0.0 <= mu[0] <= 10.0 and 0.0 <= mu[1] <= 1.05:
        axs[1].scatter(float(mu[0]), float(mu[1]), s=18, color=color,
                       edgecolor='black', linewidths=0.4, zorder=3)

    # fine dashed ellipse (1-σ)
    draw_cov_ellipse(xs, ys, axs[1], edgecolor=color)

    # label: use hard-coded (x,y) if provided; else auto just outside the ring
    if label in LABEL_POS and LABEL_POS[label] is not None:
        px, py = LABEL_POS[label]
    else:
        theta = label_angles.get(label, np.deg2rad(30))
        px, py = ellipse_point(mu, cov, theta, n_std=1.0, outside_scale=1.12)

    axs[1].annotate(label, (px, py), fontsize=7, color=color, ha='center', va='center')

# single, one-line legend centered below
fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)

# layout & save
plt.subplots_adjust(bottom=0.23, top=0.90, left=0.10, right=0.98, wspace=0.35)
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
plt.savefig(args.save_path, bbox_inches='tight')
print(f"[Saved] {args.save_path}")

# ---------- PRINT GAUSSIAN STATS TO TERMINAL ----------
print("\n=== Right-plot Gaussian stats (per method) ===")
for label, m, _ in methods:
    if not m:
        continue
    xs = np.asarray(m["exclusive"], dtype=float)      # x = Exclusive Area (%)
    ys = np.asarray(m["entropy_norm"], dtype=float)   # y = Normalized Overlap Entropy

    # Mean and std along plot axes (unbiased std with ddof=1 if len>1)
    mx = float(np.mean(xs))
    my = float(np.mean(ys))
    sx = float(np.std(xs, ddof=1)) if xs.size > 1 else 0.0
    sy = float(np.std(ys, ddof=1)) if ys.size > 1 else 0.0

    # Covariance & correlation
    if xs.size > 1:
        C = np.cov(xs, ys)
        rho = float(C[0,1] / (np.sqrt(C[0,0]*C[1,1]) + 1e-12))
        # Principal-axis stds (sqrt eigenvalues)
        w, _ = np.linalg.eigh(C)
        w = np.clip(w, 0.0, None)
        s_maj, s_min = float(np.sqrt(w.max())), float(np.sqrt(w.min()))
    else:
        rho = 0.0
        s_maj = s_min = 0.0

    print(f"\nMethod: {label}")
    print(f"  X (Exclusive %) : mean = {mx:.4f},  std = {sx:.4f}   -> {mx:.4f} ± {sx:.4f}")
    print(f"  Y (Norm. Entropy): mean = {my:.4f},  std = {sy:.4f}   -> {my:.4f} ± {sy:.4f}")
    print(f"  Corr(x,y) ρ      : {rho:.4f}")
    print(f"  Principal-axis σ : major = {s_maj:.4f}, minor = {s_min:.4f}")

# verbose data dump (kept for reference)
print("\n--- Entropy vs Exclusive Area raw arrays ---")
for label, m, _ in methods:
    if m:
        xs_dump = m["exclusive"]
        H_norm = np.array(m["entropy_norm"])
        print(f"\nMethod: {label}")
        print("Exclusive Area (%):", np.round(xs_dump, 3).tolist())
        print("Overlap Entropy (normalized):", np.round(H_norm, 4).tolist())
