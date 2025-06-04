#!/usr/bin/env python3
"""
Post-processing script for simulation results analysis.
Generates plots and summary tables from multiple seed runs.
Now plots vs. time (in seconds) instead of iteration count.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_seed_data(base_dir, strategy, k_attr, k_rep):
    """Load data from all seed directories for a given strategy."""
    data = []
    seed_dirs = glob.glob(os.path.join(base_dir, f"{strategy}_seeds", "seed*"))

    for seed_dir in sorted(seed_dirs):
        seed_num = int(os.path.basename(seed_dir).replace('seed', ''))
        pattern = f"{strategy}_log_kattr{k_attr:.2f}_krep{k_rep:.2f}_seed{seed_num}_metrics.npz"
        metrics_files = glob.glob(os.path.join(seed_dir, pattern))

        if metrics_files:
            try:
                metrics = np.load(metrics_files[0])
                time = metrics['time']
                coverage = metrics['coverage']
                redundancy = metrics['redundancy']
                affinity = metrics['affinity']
                data.append({
                    'seed': seed_num,
                    'time': time,
                    'coverage': coverage,
                    'redundancy': redundancy,
                    'affinity': affinity,
                    'num_viewpoints': int(metrics['num_viewpoints']),
                    'num_iterations': int(metrics['num_iterations']),
                })
            except Exception as e:
                print(f"Warning: Could not load {metrics_files[0]}: {e}")
        else:
            print(f"Warning: No metrics file found in {seed_dir}")

    return data

def compute_auc(values, times):
    """Compute Area Under Curve using trapezoidal rule."""
    if len(values) < 2 or len(times) != len(values):
        return 0.0
    return np.trapz(values, x=times)

def compute_summary_stats(data_list, strategy):
    """Compute summary statistics from list of simulation data."""
    if not data_list:
        return None

    final_coverage = [d['coverage'][-1] for d in data_list]
    final_redundancy = [d['redundancy'][-1] for d in data_list]
    final_affinity = [d['affinity'][-1] for d in data_list]
    num_viewpoints = [d['num_viewpoints'] for d in data_list]
    num_iterations = [d['num_iterations'] for d in data_list]
    simulation_time = [d['time'][-1] for d in data_list]  # Take last time value

    coverage_aucs = [compute_auc(d['coverage'], d['time']) for d in data_list]
    redundancy_aucs = [compute_auc(d['redundancy'], d['time']) for d in data_list]
    affinity_aucs = [compute_auc(d['affinity'], d['time']) for d in data_list]

    def safe_stats(values):
        if not values:
            return 0.0, 0.0
        arr = np.array(values)
        return np.mean(arr), np.std(arr)

    stats = {
        'strategy': strategy,
        'n_seeds': len(data_list),
        'final_coverage_mean': safe_stats(final_coverage)[0],
        'final_coverage_std': safe_stats(final_coverage)[1],
        'final_redundancy_mean': safe_stats(final_redundancy)[0],
        'final_redundancy_std': safe_stats(final_redundancy)[1],
        'final_affinity_mean': safe_stats(final_affinity)[0],
        'final_affinity_std': safe_stats(final_affinity)[1],
        'num_viewpoints_mean': safe_stats(num_viewpoints)[0],
        'num_viewpoints_std': safe_stats(num_viewpoints)[1],
        'num_iterations_mean': safe_stats(num_iterations)[0],
        'num_iterations_std': safe_stats(num_iterations)[1],
        'simulation_time_mean': safe_stats(simulation_time)[0],
        'simulation_time_std': safe_stats(simulation_time)[1],
        'coverage_auc_mean': safe_stats(coverage_aucs)[0],
        'coverage_auc_std': safe_stats(coverage_aucs)[1],
        'redundancy_auc_mean': safe_stats(redundancy_aucs)[0],
        'redundancy_auc_std': safe_stats(redundancy_aucs)[1],
        'affinity_auc_mean': safe_stats(affinity_aucs)[0],
        'affinity_auc_std': safe_stats(affinity_aucs)[1],
    }

    return stats

def create_time_plots(nbv_data, random_data, save_dir):
    """Create plots vs. time instead of iterations."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Simulation Results Over Time (NBV vs Random)', fontsize=16)

    for label, data, color in [('NBV', nbv_data, 'blue'), ('Random', random_data, 'red')]:
        if not data:
            continue

        common_time = np.linspace(0, max(max(d['time']) for d in data), 500)
        cov_interp = np.array([np.interp(common_time, d['time'], d['coverage']) for d in data])
        red_interp = np.array([np.interp(common_time, d['time'], d['redundancy']) for d in data])
        aff_interp = np.array([np.interp(common_time, d['time'], d['affinity']) for d in data])

        cov_mean, cov_std = np.mean(cov_interp, axis=0), np.std(cov_interp, axis=0)
        red_mean, red_std = np.mean(red_interp, axis=0), np.std(red_interp, axis=0)
        aff_mean, aff_std = np.mean(aff_interp, axis=0), np.std(aff_interp, axis=0)

        axes[0, 0].plot(common_time, cov_mean, label=f'{label} (n={len(data)})', color=color)
        axes[0, 0].fill_between(common_time, cov_mean - cov_std, cov_mean + cov_std, alpha=0.3, color=color)

        axes[0, 1].plot(common_time, red_mean, label=f'{label} (n={len(data)})', color=color)
        axes[0, 1].fill_between(common_time, red_mean - red_std, red_mean + red_std, alpha=0.3, color=color)

        axes[1, 0].plot(common_time, aff_mean, label=f'{label} (n={len(data)})', color=color)
        axes[1, 0].fill_between(common_time, aff_mean - aff_std, aff_mean + aff_std, alpha=0.3, color=color)

        final_metrics = [cov_interp[:, -1], red_interp[:, -1]]
        labels = [f'{label} Coverage', f'{label} Redundancy']
        colors = ['lightblue' if label == 'NBV' else 'lightcoral'] * 2
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
    path = os.path.join(save_dir, 'comparison_over_time.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {path}")

def create_summary_table(nbv_stats, random_stats, save_dir):
    """Create comprehensive summary table."""
    rows = []
    if nbv_stats:
        rows.append(nbv_stats)
    if random_stats:
        rows.append(random_stats)
    if not rows:
        print("No data to summarize.")
        return

    df = pd.DataFrame(rows)
    column_order = [k for k in df.columns if k != 'strategy']
    df = df[['strategy'] + column_order]
    df[df.select_dtypes(include=[np.number]).columns] = df.select_dtypes(include=[np.number]).round(4)
    path = os.path.join(save_dir, 'summary_statistics.csv')
    df.to_csv(path, index=False)
    print(f"\nSaved CSV summary: {path}")

    for _, row in df.iterrows():
        print(f"\n{row['strategy'].upper()} (n={int(row['n_seeds'])} seeds):")
        print(f"Coverage:   {row['final_coverage_mean']:.3f} ± {row['final_coverage_std']:.3f}")
        print(f"Redundancy: {row['final_redundancy_mean']:.3f} ± {row['final_redundancy_std']:.3f}")
        print(f"Affinity:   {row['final_affinity_mean']:.3f} ± {row['final_affinity_std']:.3f}")
        print(f"AUC Cov:    {row['coverage_auc_mean']:.1f} ± {row['coverage_auc_std']:.1f}")
        print(f"AUC Red:    {row['redundancy_auc_mean']:.1f} ± {row['redundancy_auc_std']:.1f}")
        print(f"AUC Aff:    {row['affinity_auc_mean']:.1f} ± {row['affinity_auc_std']:.1f}")
        print(f"Views:      {row['num_viewpoints_mean']:.1f} ± {row['num_viewpoints_std']:.1f}")
        print(f"Iterations: {row['num_iterations_mean']:.1f} ± {row['num_iterations_std']:.1f}")
        print(f"Time:       {row['simulation_time_mean']:.2f} ± {row['simulation_time_std']:.2f} sec")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='./results2')
    parser.add_argument('--k_attr', type=float, default=7.0)
    parser.add_argument('--k_rep', type=float, default=6.0)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    args.output_dir = args.output_dir or args.results_dir
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    nbv_data = load_seed_data(args.results_dir, 'nbv', args.k_attr, args.k_rep)
    rand_data = load_seed_data(args.results_dir, 'random', args.k_attr, args.k_rep)

    print("Computing stats...")
    nbv_stats = compute_summary_stats(nbv_data, 'nbv') if nbv_data else None
    rand_stats = compute_summary_stats(rand_data, 'random') if rand_data else None

    print("Creating plots...")
    create_time_plots(nbv_data, rand_data, args.output_dir)

    print("Creating tables...")
    create_summary_table(nbv_stats, rand_stats, args.output_dir)

    print("Done.")

if __name__ == "__main__":
    main()
