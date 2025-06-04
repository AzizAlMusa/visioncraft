#!/usr/bin/env python3
"""
Unified post-processing for NBV, Random, Greedy, Genetic, and SA.
Generates:
- summary_statistics.csv
- comparison_all_methods.png (Coverage, Redundancy, Affinity vs Time)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

def load_all_data(configs):
    data_by_strategy = {}
    for config in configs:
        strategy = config['strategy']
        file_prefix = config.get('file_prefix', strategy)
        pattern = config.get('pattern', f"{file_prefix}_seed{{}}_metrics.npz")
        base_dir = config['dir']
        data = []
        seed_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("seed")])

        for seed_dir in seed_dirs:
            seed_num = int(seed_dir.replace('seed', ''))
            full_path = os.path.join(base_dir, seed_dir, pattern.format(seed_num))
            if os.path.exists(full_path):
                try:
                    metrics = np.load(full_path)
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
                    print(f"Warning: Could not load {full_path}: {e}")
            else:
                print(f"Warning: Missing {full_path}")
        data_by_strategy[strategy] = data
    return data_by_strategy

def compute_auc(values, times):
    if len(values) < 2 or len(times) != len(values):
        return 0.0
    return np.trapz(values, x=times)

def compute_summary(data_by_strategy):
    summaries = []
    for strategy, data_list in data_by_strategy.items():
        if not data_list:
            continue
        def stats(values):
            arr = np.array(values)
            return np.mean(arr), np.std(arr)
        final_coverage = [d['coverage'][-1] for d in data_list]
        final_redundancy = [d['redundancy'][-1] for d in data_list]
        final_affinity = [d['affinity'][-1] for d in data_list]
        num_viewpoints = [d['num_viewpoints'] for d in data_list]
        num_iterations = [d['num_iterations'] for d in data_list]
        sim_time = [d['time'][-1] for d in data_list]
        coverage_auc = [compute_auc(d['coverage'], d['time']) for d in data_list]
        redundancy_auc = [compute_auc(d['redundancy'], d['time']) for d in data_list]
        affinity_auc = [compute_auc(d['affinity'], d['time']) for d in data_list]

        summary = {
            'strategy': strategy,
            'n_seeds': len(data_list),
            'final_coverage_mean': stats(final_coverage)[0],
            'final_coverage_std': stats(final_coverage)[1],
            'final_redundancy_mean': stats(final_redundancy)[0],
            'final_redundancy_std': stats(final_redundancy)[1],
            'final_affinity_mean': stats(final_affinity)[0],
            'final_affinity_std': stats(final_affinity)[1],
            'num_viewpoints_mean': stats(num_viewpoints)[0],
            'num_viewpoints_std': stats(num_viewpoints)[1],
            'num_iterations_mean': stats(num_iterations)[0],
            'num_iterations_std': stats(num_iterations)[1],
            'simulation_time_mean': stats(sim_time)[0],
            'simulation_time_std': stats(sim_time)[1],
            'coverage_auc_mean': stats(coverage_auc)[0],
            'coverage_auc_std': stats(coverage_auc)[1],
            'redundancy_auc_mean': stats(redundancy_auc)[0],
            'redundancy_auc_std': stats(redundancy_auc)[1],
            'affinity_auc_mean': stats(affinity_auc)[0],
            'affinity_auc_std': stats(affinity_auc)[1],
        }
        summaries.append(summary)
    return pd.DataFrame(summaries)

def plot_comparative_curves(data_by_strategy, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    colors = {
        'nbv': 'blue', 'random': 'red', 'greedy': 'purple',
        'genetic': 'green', 'sa': 'orange'
    }

    for strategy, data in data_by_strategy.items():
        if not data: continue
        color = colors.get(strategy, 'gray')
        label = strategy.upper()
        common_time = np.linspace(0, max(max(d['time']) for d in data), 500)
        cov_interp = np.array([np.interp(common_time, d['time'], d['coverage']) for d in data])
        red_interp = np.array([np.interp(common_time, d['time'], d['redundancy']) for d in data])
        aff_interp = np.array([np.interp(common_time, d['time'], d['affinity']) for d in data])
        cov_mean, cov_std = np.mean(cov_interp, axis=0), np.std(cov_interp, axis=0)
        red_mean, red_std = np.mean(red_interp, axis=0), np.std(red_interp, axis=0)
        aff_mean, aff_std = np.mean(aff_interp, axis=0), np.std(aff_interp, axis=0)

        axes[0, 0].plot(common_time, cov_mean, label=label, color=color)
        axes[0, 0].fill_between(common_time, cov_mean - cov_std, cov_mean + cov_std, alpha=0.3, color=color)
        axes[0, 1].plot(common_time, red_mean, label=label, color=color)
        axes[0, 1].fill_between(common_time, red_mean - red_std, red_mean + red_std, alpha=0.3, color=color)
        axes[1, 0].plot(common_time, aff_mean, label=label, color=color)
        axes[1, 0].fill_between(common_time, aff_mean - aff_std, aff_mean + aff_std, alpha=0.3, color=color)

    for ax, title in zip(axes.flat[:3], ['Coverage', 'Redundancy', 'Affinity']):
        ax.set_title(f'{title} Over Time')
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[1, 1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='./results2')
    parser.add_argument('--k_attr', type=float, default=10.0)
    parser.add_argument('--k_rep', type=float, default=0.25)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    # Folder and file pattern config
    configs = [
        {'strategy': 'nbv', 'dir': os.path.join(args.results_dir, 'nbv_seeds'),
         'pattern': f'nbv_log_kattr{args.k_attr:.2f}_krep{args.k_rep:.2f}_seed{{}}_metrics.npz'},
        {'strategy': 'random', 'dir': os.path.join(args.results_dir, 'random_seeds'),
         'pattern': f'random_log_kattr{args.k_attr:.2f}_krep{args.k_rep:.2f}_seed{{}}_metrics.npz'},
        {'strategy': 'greedy', 'dir': os.path.join(args.results_dir, 'greedy_seeds'),
         'pattern': 'greedy_seed{}_metrics.npz'},
        {'strategy': 'genetic', 'dir': os.path.join(args.results_dir, 'genetic_seeds'),
         'pattern': 'rkga_seed{}_metrics.npz'},
        {'strategy': 'sa', 'dir': os.path.join(args.results_dir, 'sa_seeds'),
         'pattern': 'sa_seed{}_metrics.npz'},
    ]

    print("Loading data...")
    data_by_strategy = load_all_data(configs)

    print("Computing summary statistics...")
    summary_df = compute_summary(data_by_strategy)
    summary_path = os.path.join(output_dir, 'summary_statistics.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary table: {summary_path}")

    print("Generating comparison plot...")
    plot_path = os.path.join(output_dir, 'comparison_all_methods.png')
    plot_comparative_curves(data_by_strategy, plot_path)

    print("Done.")

if __name__ == '__main__':
    main()
