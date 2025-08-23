#!/usr/bin/env python3
"""
Post-processing script for 3D Greedy simulation results.
Supports both layouts:
1) ./greedy/seed_0, seed_1, ...
2) ./greedy/<model>/seed_0, seed_1, ...
Saves each model's outputs into that model's directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import argparse
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# Core utilities
# ------------------------------

def load_seed_data(base_dir, strategy):
    """Load data from all seed_* directories under base_dir."""
    data = []
    seed_dirs = glob.glob(os.path.join(base_dir, "seed_*"))
    seed_dirs = sorted(seed_dirs, key=lambda p: int(os.path.basename(p).split('_')[-1])
                       if os.path.basename(p).split('_')[-1].isdigit() else 1_000_000)

    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)
        # Expect: greedy_result.npz
        pattern = f"{strategy}_result.npz"
        metrics_files = glob.glob(os.path.join(seed_dir, pattern))
        if not metrics_files:
            print(f"[warn] No metrics file found in {seed_dir}")
            continue

        path = metrics_files[0]
        try:
            metrics = np.load(path)
            data.append({
                'seed': int(seed_name.split('_')[-1]) if seed_name.split('_')[-1].isdigit() else -1,
                'time': metrics['time'],
                'coverage': metrics['coverage'],
                'redundancy': metrics['redundancy'],
                'affinity': metrics['affinity'],
                'num_viewpoints': int(metrics['num_viewpoints']),
                'num_iterations': int(metrics['num_iterations']),
            })
        except Exception as e:
            print(f"[warn] Could not load {path}: {e}")

    return data

def compute_auc(values, times):
    """Compute Area Under Curve using trapezoidal rule."""
    if len(values) < 2 or len(times) != len(values):
        return 0.0
    return float(np.trapz(values, x=times))

def compute_summary_stats(data_list, strategy):
    """Compute summary statistics from list of simulation data."""
    if not data_list:
        return None

    def mean_std(vals):
        arr = np.array(vals, dtype=float)
        return float(np.mean(arr)), float(np.std(arr))

    final_coverage = [d['coverage'][-1] for d in data_list]
    final_redundancy = [d['redundancy'][-1] for d in data_list]
    final_affinity = [d['affinity'][-1] for d in data_list]
    num_viewpoints = [d['num_viewpoints'] for d in data_list]
    num_iterations = [d['num_iterations'] for d in data_list]
    simulation_time = [d['time'][-1] for d in data_list]

    coverage_aucs = [compute_auc(d['coverage'], d['time']) for d in data_list]
    redundancy_aucs = [compute_auc(d['redundancy'], d['time']) for d in data_list]
    affinity_aucs = [compute_auc(d['affinity'], d['time']) for d in data_list]

    stats = {
        'strategy': strategy,
        'n_seeds': len(data_list),
        'final_coverage_mean': mean_std(final_coverage)[0],
        'final_coverage_std':  mean_std(final_coverage)[1],
        'final_redundancy_mean': mean_std(final_redundancy)[0],
        'final_redundancy_std':  mean_std(final_redundancy)[1],
        'final_affinity_mean': mean_std(final_affinity)[0],
        'final_affinity_std':  mean_std(final_affinity)[1],
        'num_viewpoints_mean': mean_std(num_viewpoints)[0],
        'num_viewpoints_std':  mean_std(num_viewpoints)[1],
        'num_iterations_mean': mean_std(num_iterations)[0],
        'num_iterations_std':  mean_std(num_iterations)[1],
        'simulation_time_mean': mean_std(simulation_time)[0],
        'simulation_time_std':  mean_std(simulation_time)[1],
        'coverage_auc_mean': mean_std(coverage_aucs)[0],
        'coverage_auc_std':  mean_std(coverage_aucs)[1],
        'redundancy_auc_mean': mean_std(redundancy_aucs)[0],
        'redundancy_auc_std':  mean_std(redundancy_aucs)[1],
        'affinity_auc_mean': mean_std(affinity_aucs)[0],
        'affinity_auc_std':  mean_std(affinity_aucs)[1],
    }
    return stats

def create_time_plots(greedy_data, save_dir, title_suffix=""):
    """Create plots vs. time for greedy data and save to save_dir."""
    if not greedy_data:
        print("[info] No data to plot.")
        return

    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Simulation Results Over Time (Greedy){title_suffix}', fontsize=16)

    label = 'Greedy'
    color = 'purple'
    # Build a common time axis across all seeds
    max_t = max(float(np.max(d['time'])) for d in greedy_data)
    common_time = np.linspace(0.0, max_t, 500)

    # Interpolate each curve on common_time
    cov_interp = np.array([np.interp(common_time, d['time'], d['coverage']) for d in greedy_data])
    red_interp = np.array([np.interp(common_time, d['time'], d['redundancy']) for d in greedy_data])
    aff_interp = np.array([np.interp(common_time, d['time'], d['affinity']) for d in greedy_data])

    cov_mean, cov_std = np.mean(cov_interp, axis=0), np.std(cov_interp, axis=0)
    red_mean, red_std = np.mean(red_interp, axis=0), np.std(red_interp, axis=0)
    aff_mean, aff_std = np.mean(aff_interp, axis=0), np.std(aff_interp, axis=0)

    axes[0, 0].plot(common_time, cov_mean, label=f'{label} (n={len(greedy_data)})', color=color)
    axes[0, 0].fill_between(common_time, cov_mean - cov_std, cov_mean + cov_std, alpha=0.3, color=color)

    axes[0, 1].plot(common_time, red_mean, label=f'{label} (n={len(greedy_data)})', color=color)
    axes[0, 1].fill_between(common_time, red_mean - red_std, red_mean + red_std, alpha=0.3, color=color)

    axes[1, 0].plot(common_time, aff_mean, label=f'{label} (n={len(greedy_data)})', color=color)
    axes[1, 0].fill_between(common_time, aff_mean - aff_std, aff_mean + aff_std, alpha=0.3, color=color)

    final_metrics = [cov_interp[:, -1], red_interp[:, -1]]
    labels = [f'{label} Coverage', f'{label} Redundancy']
    colors = ['lightblue', 'lightblue']
    bp = axes[1, 1].boxplot(final_metrics, labels=labels, patch_artist=True)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)

    axes[0, 0].set_title('Coverage Over Time')
    axes[0, 1].set_title('Redundancy Over Time')
    axes[1, 0].set_title('Affinity Over Time')
    axes[1, 1].set_title('Final Metrics Distribution')

    for ax in axes.flat:
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)

    axes[0, 0].legend()
    plt.tight_layout()
    path = os.path.join(save_dir, 'greedy_over_time.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot: {path}")

def create_summary_table(stats, save_dir):
    """Create CSV and print formatted summary in save_dir."""
    if not stats:
        print("[info] No data to summarize.")
        return

    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame([stats])
    column_order = [k for k in df.columns if k != 'strategy']
    df = df[['strategy'] + column_order]
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].round(4)

    path = os.path.join(save_dir, 'summary_statistics.csv')
    df.to_csv(path, index=False)
    print(f"Saved CSV summary: {path}")

    print(f"\n{stats['strategy'].upper()} (n={int(stats['n_seeds'])} seeds):")
    print(f"Coverage:   {stats['final_coverage_mean']:.3f} ± {stats['final_coverage_std']:.3f}")
    print(f"Redundancy: {stats['final_redundancy_mean']:.3f} ± {stats['final_redundancy_std']:.3f}")
    print(f"Affinity:   {stats['final_affinity_mean']:.3f} ± {stats['final_affinity_std']:.3f}")
    print(f"AUC Cov:    {stats['coverage_auc_mean']:.1f} ± {stats['coverage_auc_std']:.1f}")
    print(f"AUC Red:    {stats['redundancy_auc_mean']:.1f} ± {stats['redundancy_auc_std']:.1f}")
    print(f"AUC Aff:    {stats['affinity_auc_mean']:.1f} ± {stats['affinity_auc_std']:.1f}")
    print(f"Views:      {stats['num_viewpoints_mean']:.1f} ± {stats['num_viewpoints_std']:.1f}")
    print(f"Iterations: {stats['num_iterations_mean']:.1f} ± {stats['num_iterations_std']:.1f}")
    print(f"Time:       {stats['simulation_time_mean']:.2f} ± {stats['simulation_time_std']:.2f} sec\n")

# ------------------------------
# Layout detection and processing
# ------------------------------

def contains_seeds(dir_path):
    """Return True if dir_path has any seed_* subfolders."""
    return any(os.path.isdir(p) for p in glob.glob(os.path.join(dir_path, "seed_*")))

def list_model_dirs(results_dir):
    """
    If results_dir directly contains seed_* => old layout: treat results_dir as a single 'model' dir.
    Else => new layout: return subdirectories under results_dir that contain seed_*.
    """
    if contains_seeds(results_dir):
        return [results_dir]  # single 'model' (old layout)
    # Otherwise, look for subdirectories that contain seed_*
    candidates = [p for p in glob.glob(os.path.join(results_dir, "*")) if os.path.isdir(p)]
    model_dirs = [p for p in candidates if contains_seeds(p)]
    return sorted(model_dirs)

def model_name_from_dir(model_dir, results_dir):
    """Derive a readable model name from the directory path."""
    # Prefer leaf directory name under results_dir
    try:
        rel = os.path.relpath(model_dir, results_dir)
        leaf = rel.split(os.sep)[0]
        return leaf if leaf not in (".", "") else os.path.basename(model_dir.rstrip(os.sep))
    except Exception:
        return os.path.basename(model_dir.rstrip(os.sep))

def process_one_model(model_dir, strategy="greedy", title_suffix=""):
    """Load, compute, and write outputs for one model directory."""
    data = load_seed_data(model_dir, strategy)
    if not data:
        print(f"[info] No data found in {model_dir}. Skipping.")
        return

    stats = compute_summary_stats(data, strategy)
    # save_dir is the model_dir itself per your requirement
    save_dir = model_dir
    create_time_plots(data, save_dir, title_suffix=title_suffix)
    create_summary_table(stats, save_dir)

# ------------------------------
# CLI
# ------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='./greedy',
                        help="Path to results root. "
                             "Supports ./greedy/seed_* or ./greedy/<model>/seed_*.")
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    print(f"[info] Scanning: {results_dir}")

    model_dirs = list_model_dirs(results_dir)
    if not model_dirs:
        print("[info] No model or seed directories found.")
        return

    # Process each model independently; outputs go into each model's directory
    for model_dir in model_dirs:
        name = model_name_from_dir(model_dir, results_dir)
        print(f"[info] Processing model: {name} ({model_dir})")
        process_one_model(model_dir, strategy="greedy", title_suffix=f" — {name}")

    print("[info] Done.")

if __name__ == "__main__":
    main()
