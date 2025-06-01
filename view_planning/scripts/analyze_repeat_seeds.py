#!/usr/bin/env python3
"""
Post-processing script for simulation results analysis.
Generates plots and summary tables from multiple seed runs.
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
        # Extract seed number from directory name
        seed_num = int(os.path.basename(seed_dir).replace('seed', ''))
        
        # Look for the metrics file
        pattern = f"{strategy}_log_kattr{k_attr:.2f}_krep{k_rep:.2f}_seed{seed_num}_metrics.npz"
        metrics_files = glob.glob(os.path.join(seed_dir, pattern))
        
        if metrics_files:
            try:
                metrics = np.load(metrics_files[0])
                data.append({
                    'seed': seed_num,
                    'coverage': metrics['coverage'],
                    'redundancy': metrics['redundancy'],
                    'affinity': metrics['affinity'],
                    'num_viewpoints': int(metrics['num_viewpoints']),
                    'num_iterations': int(metrics['num_iterations']),
                    'simulation_time_sec': float(metrics['simulation_time_sec'])
                })
            except Exception as e:
                print(f"Warning: Could not load {metrics_files[0]}: {e}")
        else:
            print(f"Warning: No metrics file found in {seed_dir}")
    
    return data

def compute_auc(values):
    """Compute Area Under Curve using trapezoidal rule."""
    if len(values) < 2:
        return 0.0
    return np.trapz(values, dx=1.0)

def compute_summary_stats(data_list, strategy):
    """Compute summary statistics from list of simulation data."""
    if not data_list:
        return None
    
    # Extract final values
    final_coverage = [d['coverage'][-1] if len(d['coverage']) > 0 else 0.0 for d in data_list]
    final_redundancy = [d['redundancy'][-1] if len(d['redundancy']) > 0 else 0.0 for d in data_list]
    final_affinity = [d['affinity'][-1] if len(d['affinity']) > 0 else 0.0 for d in data_list]
    
    num_viewpoints = [d['num_viewpoints'] for d in data_list]
    num_iterations = [d['num_iterations'] for d in data_list]
    simulation_time = [d['simulation_time_sec'] for d in data_list]
    
    # Compute AUCs
    coverage_aucs = [compute_auc(d['coverage']) for d in data_list]
    redundancy_aucs = [compute_auc(d['redundancy']) for d in data_list]
    affinity_aucs = [compute_auc(d['affinity']) for d in data_list]
    
    def safe_stats(values):
        """Compute mean and std, handling empty lists."""
        if not values:
            return 0.0, 0.0
        arr = np.array(values)
        return np.mean(arr), np.std(arr)
    
    # Compute statistics
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

def align_sequences(data_list, max_length=None):
    """Align sequences to the same length for averaging."""
    if not data_list:
        return np.array([]), np.array([]), np.array([])
    
    # Find maximum length if not specified
    if max_length is None:
        max_length = max(len(d['coverage']) for d in data_list)
    
    aligned_coverage = []
    aligned_redundancy = []
    aligned_affinity = []
    
    for d in data_list:
        # Pad or truncate sequences
        coverage = d['coverage'][:max_length]
        redundancy = d['redundancy'][:max_length]
        affinity = d['affinity'][:max_length]
        
        if len(coverage) < max_length:
            # Pad with last value
            last_cov = coverage[-1] if len(coverage) > 0 else 0.0
            last_red = redundancy[-1] if len(redundancy) > 0 else 0.0
            last_aff = affinity[-1] if len(affinity) > 0 else 0.0
            
            coverage = np.pad(coverage, (0, max_length - len(coverage)), 
                            mode='constant', constant_values=last_cov)
            redundancy = np.pad(redundancy, (0, max_length - len(redundancy)), 
                              mode='constant', constant_values=last_red)
            affinity = np.pad(affinity, (0, max_length - len(affinity)), 
                            mode='constant', constant_values=last_aff)
        
        aligned_coverage.append(coverage)
        aligned_redundancy.append(redundancy)
        aligned_affinity.append(affinity)
    
    return np.array(aligned_coverage), np.array(aligned_redundancy), np.array(aligned_affinity)

def create_plots(nbv_data, random_data, save_dir):
    """Create comprehensive plots comparing NBV and Random strategies."""
    
    # Align sequences for both strategies
    max_len = max(
        max(len(d['coverage']) for d in nbv_data) if nbv_data else 0,
        max(len(d['coverage']) for d in random_data) if random_data else 0
    )
    
    nbv_cov, nbv_red, nbv_aff = align_sequences(nbv_data, max_len)
    rand_cov, rand_red, rand_aff = align_sequences(random_data, max_len)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Simulation Results Comparison: NBV vs Random Strategy', fontsize=16)
    
    iterations = np.arange(max_len)
    
    # Plot 1: Coverage over time
    ax1 = axes[0, 0]
    if len(nbv_cov) > 0:
        nbv_cov_mean = np.mean(nbv_cov, axis=0)
        nbv_cov_std = np.std(nbv_cov, axis=0)
        ax1.plot(iterations, nbv_cov_mean, 'b-', label=f'NBV (n={len(nbv_cov)})', linewidth=2)
        ax1.fill_between(iterations, nbv_cov_mean - nbv_cov_std, nbv_cov_mean + nbv_cov_std, 
                        alpha=0.3, color='blue')
    
    if len(rand_cov) > 0:
        rand_cov_mean = np.mean(rand_cov, axis=0)
        rand_cov_std = np.std(rand_cov, axis=0)
        ax1.plot(iterations, rand_cov_mean, 'r-', label=f'Random (n={len(rand_cov)})', linewidth=2)
        ax1.fill_between(iterations, rand_cov_mean - rand_cov_std, rand_cov_mean + rand_cov_std, 
                        alpha=0.3, color='red')
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Coverage')
    ax1.set_title('Coverage Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Redundancy over time
    ax2 = axes[0, 1]
    if len(nbv_red) > 0:
        nbv_red_mean = np.mean(nbv_red, axis=0)
        nbv_red_std = np.std(nbv_red, axis=0)
        ax2.plot(iterations, nbv_red_mean, 'b-', label=f'NBV (n={len(nbv_red)})', linewidth=2)
        ax2.fill_between(iterations, nbv_red_mean - nbv_red_std, nbv_red_mean + nbv_red_std, 
                        alpha=0.3, color='blue')
    
    if len(rand_red) > 0:
        rand_red_mean = np.mean(rand_red, axis=0)
        rand_red_std = np.std(rand_red, axis=0)
        ax2.plot(iterations, rand_red_mean, 'r-', label=f'Random (n={len(rand_red)})', linewidth=2)
        ax2.fill_between(iterations, rand_red_mean - rand_red_std, rand_red_mean + rand_red_std, 
                        alpha=0.3, color='red')
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Redundancy')
    ax2.set_title('Redundancy Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Affinity over time
    ax3 = axes[1, 0]
    if len(nbv_aff) > 0:
        nbv_aff_mean = np.mean(nbv_aff, axis=0)
        nbv_aff_std = np.std(nbv_aff, axis=0)
        ax3.plot(iterations, nbv_aff_mean, 'b-', label=f'NBV (n={len(nbv_aff)})', linewidth=2)
        ax3.fill_between(iterations, nbv_aff_mean - nbv_aff_std, nbv_aff_mean + nbv_aff_std, 
                        alpha=0.3, color='blue')
    
    if len(rand_aff) > 0:
        rand_aff_mean = np.mean(rand_aff, axis=0)
        rand_aff_std = np.std(rand_aff, axis=0)
        ax3.plot(iterations, rand_aff_mean, 'r-', label=f'Random (n={len(rand_aff)})', linewidth=2)
        ax3.fill_between(iterations, rand_aff_mean - rand_aff_std, rand_aff_mean + rand_aff_std, 
                        alpha=0.3, color='red')
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Affinity')
    ax3.set_title('Affinity Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Final metrics comparison (box plot)
    ax4 = axes[1, 1]
    final_metrics = []
    labels = []
    colors = []
    
    if len(nbv_cov) > 0:
        final_metrics.extend([nbv_cov[:, -1], nbv_red[:, -1]])
        labels.extend(['NBV Coverage', 'NBV Redundancy'])
        colors.extend(['lightblue', 'lightblue'])
    
    if len(rand_cov) > 0:
        final_metrics.extend([rand_cov[:, -1], rand_red[:, -1]])
        labels.extend(['Random Coverage', 'Random Redundancy'])
        colors.extend(['lightcoral', 'lightcoral'])
    
    if final_metrics:
        bp = ax4.boxplot(final_metrics, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
    
    ax4.set_title('Final Metrics Distribution')
    ax4.set_ylabel('Value')
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_dir, 'comparison_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to: {plot_path}")
    return plot_path

def create_summary_table(nbv_stats, random_stats, save_dir):
    """Create comprehensive summary table."""
    
    # Create DataFrame
    rows = []
    
    if nbv_stats:
        rows.append(nbv_stats)
    if random_stats:
        rows.append(random_stats)
    
    if not rows:
        print("No data to create summary table")
        return None
    
    df = pd.DataFrame(rows)
    
    # Reorder columns for better readability
    column_order = [
        'strategy', 'n_seeds',
        'final_coverage_mean', 'final_coverage_std',
        'final_redundancy_mean', 'final_redundancy_std',
        'final_affinity_mean', 'final_affinity_std',
        'coverage_auc_mean', 'coverage_auc_std',
        'redundancy_auc_mean', 'redundancy_auc_std',
        'affinity_auc_mean', 'affinity_auc_std',
        'num_viewpoints_mean', 'num_viewpoints_std',
        'num_iterations_mean', 'num_iterations_std',
        'simulation_time_mean', 'simulation_time_std'
    ]
    
    df = df[[col for col in column_order if col in df.columns]]
    
    # Round numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = df[numerical_cols].round(4)
    
    # Save to CSV
    csv_path = os.path.join(save_dir, 'summary_statistics.csv')
    df.to_csv(csv_path, index=False)
    
    # Create a formatted table for display
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    
    for _, row in df.iterrows():
        strategy = row['strategy'].upper()
        print(f"\n{strategy} Strategy (n={int(row['n_seeds'])} seeds):")
        print("-" * 50)
        print(f"Final Coverage:    {row['final_coverage_mean']:.3f} ± {row['final_coverage_std']:.3f}")
        print(f"Final Redundancy:  {row['final_redundancy_mean']:.3f} ± {row['final_redundancy_std']:.3f}")
        print(f"Final Affinity:    {row['final_affinity_mean']:.3f} ± {row['final_affinity_std']:.3f}")
        print(f"Coverage AUC:      {row['coverage_auc_mean']:.1f} ± {row['coverage_auc_std']:.1f}")
        print(f"Redundancy AUC:    {row['redundancy_auc_mean']:.1f} ± {row['redundancy_auc_std']:.1f}")
        print(f"Affinity AUC:      {row['affinity_auc_mean']:.1f} ± {row['affinity_auc_std']:.1f}")
        print(f"Num Viewpoints:    {row['num_viewpoints_mean']:.1f} ± {row['num_viewpoints_std']:.1f}")
        print(f"Num Iterations:    {row['num_iterations_mean']:.1f} ± {row['num_iterations_std']:.1f}")
        print(f"Simulation Time:   {row['simulation_time_mean']:.2f} ± {row['simulation_time_std']:.2f} seconds")
    
    # Statistical comparison if both strategies present
    if len(df) == 2:
        print(f"\n{'COMPARISON'}")
        print("-" * 50)
        nbv_row = df[df['strategy'] == 'nbv'].iloc[0] if 'nbv' in df['strategy'].values else None
        rand_row = df[df['strategy'] == 'random'].iloc[0] if 'random' in df['strategy'].values else None
        
        if nbv_row is not None and rand_row is not None:
            cov_diff = nbv_row['final_coverage_mean'] - rand_row['final_coverage_mean']
            red_diff = nbv_row['final_redundancy_mean'] - rand_row['final_redundancy_mean']
            aff_diff = nbv_row['final_affinity_mean'] - rand_row['final_affinity_mean']
            time_diff = nbv_row['simulation_time_mean'] - rand_row['simulation_time_mean']
            
            print(f"Coverage difference (NBV - Random):   {cov_diff:+.3f}")
            print(f"Redundancy difference (NBV - Random): {red_diff:+.3f}")
            print(f"Affinity difference (NBV - Random):   {aff_diff:+.3f}")
            print(f"Time difference (NBV - Random):       {time_diff:+.2f} seconds")
    
    print(f"\nDetailed statistics saved to: {csv_path}")
    return csv_path

def main():
    parser = argparse.ArgumentParser(description='Post-process simulation results')
    parser.add_argument('--results_dir', type=str, default='./results2', 
                       help='Base directory containing results')
    parser.add_argument('--k_attr', type=float, default=7.0, 
                       help='k_attr value used in simulations')
    parser.add_argument('--k_rep', type=float, default=6.0, 
                       help='k_rep value used in simulations')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots and tables (default: results_dir)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.results_dir
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading simulation data...")
    
    # Load data for both strategies
    nbv_data = load_seed_data(args.results_dir, 'nbv', args.k_attr, args.k_rep)
    random_data = load_seed_data(args.results_dir, 'random', args.k_attr, args.k_rep)
    
    print(f"Loaded {len(nbv_data)} NBV runs and {len(random_data)} Random runs")
    
    if not nbv_data and not random_data:
        print("No data found! Check your results directory and parameters.")
        return
    
    # Compute summary statistics
    nbv_stats = compute_summary_stats(nbv_data, 'nbv') if nbv_data else None
    random_stats = compute_summary_stats(random_data, 'random') if random_data else None
    
    # Create plots
    print("Creating plots...")
    create_plots(nbv_data, random_data, args.output_dir)
    
    # Create summary table
    print("Creating summary table...")
    create_summary_table(nbv_stats, random_stats, args.output_dir)
    
    print(f"\nPost-processing complete! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()