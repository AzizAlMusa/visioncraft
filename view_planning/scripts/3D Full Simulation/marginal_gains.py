#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.stats import entropy, chi2
from matplotlib.patches import Polygon
from matplotlib import colors as mcolors
import os
import glob

# ---------- Utilities for directory handling ----------
def has_seeds(dir_path):
    return any(os.path.isdir(p) for p in glob.glob(os.path.join(dir_path, "seed_*")))

def list_model_dirs(method_root):
    if has_seeds(method_root):
        return [method_root]
    candidates = [p for p in glob.glob(os.path.join(method_root, "*")) if os.path.isdir(p)]
    return sorted([p for p in candidates if has_seeds(p)])

def model_name_from_dir(model_dir, method_root):
    rel = os.path.relpath(model_dir, method_root)
    leaf = rel.split(os.sep)[0]
    return leaf if leaf not in (".", "") else os.path.basename(model_dir.rstrip(os.sep))

# ---------- Color utility ----------
def darken_hex(hex_color, factor=0.55):
    rgb = np.array(mcolors.to_rgb(hex_color))
    rgb_darker = np.clip(rgb * factor, 0, 1)
    return mcolors.to_hex(rgb_darker)

# ---------- Metric computation ----------
def compute_metrics(assignments):
    if assignments.shape[0] < assignments.shape[1]:
        assignments = assignments.T

    num_points, N = assignments.shape
    N = int(N)

    seen_counts = assignments.sum(axis=1)
    redundancy_hist = np.bincount(seen_counts, minlength=10)
    redundancy_percent = redundancy_hist / max(1, redundancy_hist.sum()) * 100.0

    exclusive_mask = (seen_counts == 1)
    exclusive_contributions = np.zeros(N, dtype=float)
    if exclusive_mask.any():
        idxs = np.where(exclusive_mask)[0]
        vps = assignments[idxs].argmax(axis=1)
        np.add.at(exclusive_contributions, vps, 1.0)
    exclusive_contributions = (exclusive_contributions / max(1, num_points)) * 100.0

    overlap = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        vi = assignments[:, i].astype(bool)
        for j in range(i, N):
            vj = assignments[:, j].astype(bool)
            inter = np.logical_and(vi, vj).sum()
            union = np.logical_or(vi, vj).sum()
            overlap[i, j] = overlap[j, i] = inter / (union + 1e-12)
    np.fill_diagonal(overlap, 1.0)

    denom = np.log(max(2, N - 1))
    overlap_entropies, overlap_cvs = [], []
    for i in range(N):
        row = np.delete(overlap[i], i)
        nbrs = row[row > 1e-12]
        if nbrs.size == 0 or denom == 0.0:
            overlap_entropies.append(0.0)
        else:
            p = nbrs / (nbrs.sum() + 1e-12)
            overlap_entropies.append(float(entropy(p)) / denom)
        if nbrs.size <= 1:
            overlap_cvs.append(0.0)
        else:
            mu = float(np.mean(nbrs))
            sd = float(np.std(nbrs, ddof=0))
            overlap_cvs.append(sd / (mu + 1e-12))

    return {
        "redundancy_percent": redundancy_percent[:6],
        "exclusive": exclusive_contributions,
        "cv": overlap_cvs,
        "entropy": overlap_entropies,
    }

def aggregate_metrics(model_dir, strategy):
    npz_files = glob.glob(os.path.join(model_dir, "seed_*", f"{strategy}_result.npz"))
    all_redundancy, all_exclusive, all_cv, all_entropy = [], [], [], []
    for npz in npz_files:
        data = np.load(npz)
        assign = data["viewpoint_point_assignments"]
        metrics = compute_metrics(assign)
        all_redundancy.append(metrics["redundancy_percent"])
        all_exclusive.extend(metrics["exclusive"])
        all_cv.extend(metrics["cv"])
        all_entropy.extend(metrics["entropy"])
    if not all_redundancy:
        return None
    avg_redundancy = np.mean(all_redundancy, axis=0)
    return {
        "redundancy_percent": avg_redundancy,
        "exclusive": np.array(all_exclusive, dtype=float),
        "cv": np.array(all_cv, dtype=float),
        "entropy": np.array(all_entropy, dtype=float),
    }

# ---------- Log-aware covariance ellipse (95%) ----------
MIN_X_LOG = 1e-3
VIS_MIN_X = 1e-3

def _filter_and_logx(xs, ys, vis_min_x=VIS_MIN_X):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys) & (xs >= vis_min_x)
    if not np.any(mask):
        return None, None, None
    xs_clamped = np.clip(xs[mask], MIN_X_LOG, None)
    xt = np.log(xs_clamped)
    yt = ys[mask]
    if xt.size < 2 or yt.size < 2:
        return xt, yt, mask
    return xt, yt, mask

def _ellipse_95_from_cov(xt, yt):
    if xt is None or yt is None or xt.size < 3 or yt.size < 3:
        return None, None
    cov = np.cov(xt, yt)
    if cov.shape != (2, 2) or not np.isfinite(cov).all():
        return None, None
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    r = np.sqrt(chi2.ppf(0.95, df=2)) * np.sqrt(np.maximum(vals, 0.0))
    t = np.linspace(0, 2*np.pi, 256)
    ellipse = (vecs @ np.vstack((r[0]*np.cos(t), r[1]*np.sin(t)))).T
    ellipse[:, 0] += np.mean(xt)
    ellipse[:, 1] += np.mean(yt)
    return np.exp(ellipse[:, 0]), ellipse[:, 1]

def draw_ellipse95_and_centroid(xs, ys, ax, color, alpha=0.08):
    xt, yt, _ = _filter_and_logx(xs, ys, VIS_MIN_X)
    if xt is None:
        return None, None
    exs, eys = _ellipse_95_from_cov(xt, yt)
    if exs is not None:
        ax.add_patch(Polygon(np.c_[exs, eys], closed=True,
                             facecolor=color, edgecolor=color,
                             linewidth=1.0, alpha=alpha))
    cx = float(np.exp(np.mean(xt))) if xt.size else None
    cy = float(np.mean(yt)) if yt.size else None
    return cx, cy

# ---------- Label placement helpers ----------
# Default quadrants
DEFAULT_POS = {
    'Greedy': 'top_right',
    'SA':     'bottom_right',
    'RKGA':   'top_right',
    'PF':     'bottom_right',
}

# Per-model overrides
OVERRIDES = {
    'gorilla': {
        'SA': 'top_left',
    },
    'cat': {
        'Greedy': 'top_left',
        'SA':     'top_right',
        'RKGA':   'bottom_right',
        # PF -> default (bottom_right)
    },
    'bracket': {
        'SA': 'top_right',
    },
    'bell': {
        'RKGA': 'bottom_right',
        'SA':   'top_right',
    },
}

def quadrant_for(model_name, label):
    m = (model_name or "").lower()
    if m in OVERRIDES and label in OVERRIDES[m]:
        return OVERRIDES[m][label]
    return DEFAULT_POS.get(label, 'top_right')

def place_label(cx, cy, quadrant, y_offset_abs, x_factor=1.15):
    """Return (lx, ly, ha, va) given centroid and quadrant name. Log-x safe."""
    if quadrant == 'top_right':
        return cx * x_factor, cy + y_offset_abs, 'left',  'bottom'
    if quadrant == 'bottom_right':
        return cx * x_factor, cy - y_offset_abs, 'left',  'top'
    if quadrant == 'top_left':
        return cx / x_factor, cy + y_offset_abs, 'right', 'bottom'
    if quadrant == 'bottom_left':
        return cx / x_factor, cy - y_offset_abs, 'right', 'top'
    # Fallback
    return cx * x_factor, cy + y_offset_abs, 'left', 'bottom'

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_greedy", type=str, default="./greedy")
    parser.add_argument("--dir_genetic", type=str, default="./genetic")
    parser.add_argument("--dir_sa", type=str, default="./sa")
    parser.add_argument("--dir_potential", type=str, default="./potential_field")
    parser.add_argument("--out_dir", type=str, default="./overlap_results")
    args = parser.parse_args()

    # Styling
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['SF Pro Display', 'Helvetica Neue', 'Arial', 'DejaVu Sans']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.titleweight'] = '600'
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.15
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.labelcolor'] = '#333333'
    plt.rcParams['text.color'] = '#333333'

    # Display names
    colors = {
        'Greedy': '#6366f1',
        'RKGA':   '#ec4899',
        'SA':     '#f59e0b',
        'PF':     '#10b981',
    }
    alphas = {'Greedy': 0.7, 'RKGA': 0.7, 'SA': 0.7, 'PF': 1.0}

    model_dirs_greedy = list_model_dirs(args.dir_greedy)
    os.makedirs(args.out_dir, exist_ok=True)

    for greedy_model_dir in model_dirs_greedy:
        model_name = model_name_from_dir(greedy_model_dir, args.dir_greedy)

        genetic_model_dir = os.path.join(args.dir_genetic, model_name)
        sa_model_dir = os.path.join(args.dir_sa, model_name)
        pot_model_dir = os.path.join(args.dir_potential, model_name)

        methods = [
            ("Greedy", aggregate_metrics(greedy_model_dir,  "greedy")),
            ("RKGA",   aggregate_metrics(genetic_model_dir, "genetic")),
            ("SA",     aggregate_metrics(sa_model_dir,      "sa")),
            ("PF",     aggregate_metrics(pot_model_dir,     "potential_field")),
        ]

        if not any(m[1] for m in methods):
            print(f"[info] No data for model {model_name}, skipping.")
            continue

        fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
        fig.patch.set_facecolor('white')

        # 1) Coverage Frequency Distribution
        x = np.arange(6)
        bar_width = 0.18
        plotted = 0
        for label, m in methods:
            if m:
                axs[0].bar(x + plotted*bar_width - bar_width*1.5,
                           m["redundancy_percent"],
                           width=bar_width,
                           label=label,
                           color=colors[label],
                           alpha=alphas[label],
                           edgecolor='white',
                           linewidth=0.8)
                plotted += 1
        axs[0].set_title("Coverage Frequency Distribution", fontweight='600', pad=15, color='#1f2937')
        axs[0].set_xlabel("Times a Region is Observed", fontweight='600', color='#374151')
        axs[0].set_ylabel("% of Area", fontweight='600', color='#374151')
        axs[0].legend(frameon=True, framealpha=0.95, edgecolor='#e5e7eb', loc='upper right')
        axs[0].grid(True, alpha=0.15, linestyle='-', linewidth=0.5, color='#9ca3af')
        axs[0].set_axisbelow(True)
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['right'].set_visible(False)
        axs[0].set_facecolor('#fefefe')

        # 2) CV vs Unique Contribution (log-x)
        ax = axs[1]
        ax.set_title("CV of Overlap vs Viewpoint Unique Contribution", fontweight='600', pad=15, color='#1f2937')
        ax.set_xlabel("Unique Contribution / Viewpoint (%)", fontweight='600', color='#374151')
        ax.set_ylabel("CV of Overlaps (neighbors only)", fontweight='600', color='#374151')
        ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.5, color='#9ca3af')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#fefefe')
        ax.set_xscale('log')

        xmax_vals, y_all = [], []
        precomputed = []
        for label, m in methods:
            if not m:
                precomputed.append((label, None, None, None))
                continue
            xs_raw = np.array(m["exclusive"], dtype=float)
            ys_raw = np.array(m["cv"], dtype=float)
            mask_vis = np.isfinite(xs_raw) & np.isfinite(ys_raw) & (xs_raw >= VIS_MIN_X)
            if not np.any(mask_vis):
                precomputed.append((label, None, None, None))
                continue
            xs = xs_raw[mask_vis]
            ys = ys_raw[mask_vis]
            xs_plot = np.clip(xs, MIN_X_LOG, None)
            xmax_vals.append(np.nanmax(xs_plot))
            y_all.append(ys)
            precomputed.append((label, xs, ys, xs_plot))

        # y-range for offsets
        if y_all:
            y_concat = np.concatenate(y_all)
            y_min = float(np.nanmin(y_concat))
            y_max = float(np.nanmax(y_concat))
        else:
            y_min, y_max = 0.0, 1.0
        y_span = max(1e-6, y_max - y_min)
        y_offset_abs = 0.03 * y_span

        # Plot and annotate
        for label, xs, ys, xs_plot in precomputed:
            if xs is None:
                continue
            color = colors[label]
            alpha = alphas[label]

            ax.scatter(xs_plot, ys, s=(25 if label == "PF" else 20),
                       alpha=alpha*0.6, color=color, edgecolors='white', linewidth=0.3)

            cx, cy = draw_ellipse95_and_centroid(xs, ys, ax, color=color, alpha=0.08)
            if cx is not None and cy is not None:
                ax.scatter(cx, cy, c='#1f2937',
                           s=(90 if label == "PF" else 70)+20,
                           marker='o', edgecolors=color, linewidth=2.0, zorder=10, alpha=alpha)
                ax.scatter(cx, cy, c=color,
                           s=(90 if label == "PF" else 70),
                           marker='o', edgecolors='white', linewidth=1.2, zorder=11, alpha=alpha)

                quadrant = quadrant_for(model_name, label)
                lx, ly, ha, va = place_label(cx, cy, quadrant, y_offset_abs, x_factor=1.15)

                label_color = darken_hex(color, factor=0.55)
                label_style = {'fontweight': '700', 'fontsize': 10} if label == "PF" \
                              else {'fontweight': '600', 'fontsize': 9}

                ax.annotate(label, (lx, ly),
                            color=label_color, ha=ha, va=va,
                            alpha=min(1.0, alpha*1.05),
                            **label_style)

        xright = np.nanmax(xmax_vals) if len(xmax_vals) else MIN_X_LOG
        if not np.isfinite(xright) or xright <= MIN_X_LOG:
            xright = 1.2
        ax.set_xlim(MIN_X_LOG, xright * 1.25)

        plt.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.12, wspace=0.30)

        save_path = os.path.join(args.out_dir, f"{model_name}_overlap_comparison_cv.pdf")
        plt.savefig(save_path, facecolor='white', dpi=300)
        plt.close(fig)
        print(f"[Saved] {save_path}")

if __name__ == "__main__":
    main()
