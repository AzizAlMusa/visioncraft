#!/usr/bin/env python3
"""
Post-processing script for Simulated Annealing (SA) simulation results.
Outputs plots and tables consistent with other strategies (greedy, rkga).
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import argparse
import warnings
warnings.filterwarnings('ignore')

def load_seed_data(base_dir, strategy, file_prefix):
    """Load data from all seed directories for a given strategy and prefix."""
    data = []
    seed_dirs = glob.glob(os.path.join(base_dir, f"{strategy}_seeds", "seed*"))

    for seed_dir in sorted(seed_dirs):
        seed_num = int(os.path.basename(seed_dir).replace('seed', ''))
        pattern = f"{file_prefix}_seed{seed_num}_metrics.npz"
        metrics_files = glob.glob(os.path.join(seed_dir, pattern))

        if metrics_files:
            try:
                metrics = np.load(metrics_files[0])
                data.append({
                    'seed': seed_num,
                    'time': metrics['time'],
                    'coverage': metrics['coverage'],
                    'redundancy': metrics['redundancy'],
                    'affinity': metrics['affinity'],
                    'num_viewpoints': int(metrics['num_viewpoints']),
                    'num_iterations': int(metrics['num_iterations']),
                })
            except Exception as e:
                print(f"Warning: Could not load {metrics_files[0]}: {e}")
        else:
            print(f"Warning: No metrics file found in {seed_dir}")

    return data

def compute_auc(values, times):
    if len(values) < 2 or len(times) != len(values):
        return 0.0
    return np.trapz(values, x=times)

def compute_summary_stats(data_list, strategy):
    if not data_list:
        return None

    def safe_stats(values):
        arr = np.array(values)
        return np.mean(arr), np.std(arr)

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

def create_time_plots(data, save_dir, label='SA', color='orange'):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Simulation Results Over Time ({label})', fontsize=16)

    if not data:
        print("No data to plot.")
        return

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
    colors = ['moccasin', 'moccasin']
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
    path = os.path.join(save_dir, 'sa_over_time.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {path}")

def create_summary_table(stats, save_dir):
    if not stats:
        print("No data to summarize.")
        return

    df = pd.DataFrame([stats])
    column_order = [k for k in df.columns if k != 'strategy']
    df = df[['strategy'] + column_order]
    df[df.select_dtypes(include=[np.number]).columns] = df.select_dtypes(include=[np.number]).round(4)

    path = os.path.join(save_dir, 'summary_statistics.csv')
    df.to_csv(path, index=False)
    print(f"\nSaved CSV summary: {path}")

    print(f"\n{stats['strategy'].upper()} (n={int(stats['n_seeds'])} seeds):")
    print(f"Coverage:   {stats['final_coverage_mean']:.3f} ± {stats['final_coverage_std']:.3f}")
    print(f"Redundancy: {stats['final_redundancy_mean']:.3f} ± {stats['final_redundancy_std']:.3f}")
    print(f"Affinity:   {stats['final_affinity_mean']:.3f} ± {stats['final_affinity_std']:.3f}")
    print(f"AUC Cov:    {stats['coverage_auc_mean']:.1f} ± {stats['coverage_auc_std']:.1f}")
    print(f"AUC Red:    {stats['redundancy_auc_mean']:.1f} ± {stats['redundancy_auc_std']:.1f}")
    print(f"AUC Aff:    {stats['affinity_auc_mean']:.1f} ± {stats['affinity_auc_std']:.1f}")
    print(f"Views:      {stats['num_viewpoints_mean']:.1f} ± {stats['num_viewpoints_std']:.1f}")
    print(f"Iterations: {stats['num_iterations_mean']:.1f} ± {stats['num_iterations_std']:.1f}")
    print(f"Time:       {stats['simulation_time_mean']:.2f} ± {stats['simulation_time_std']:.2f} sec")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='./results2')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--strategy', type=str, default='sa')
    parser.add_argument('--file_prefix', type=str, default='sa')
    args = parser.parse_args()

    args.output_dir = args.output_dir or args.results_dir
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    data = load_seed_data(args.results_dir, args.strategy, args.file_prefix)

    print("Computing stats...")
    stats = compute_summary_stats(data, args.strategy) if data else None

    print("Creating plots...")
    create_time_plots(data, args.output_dir, label=args.strategy.upper(), color='orange')

    print("Creating tables...")
    create_summary_table(stats, args.output_dir)

    print("Done.")

if __name__ == "__main__":
    main()
