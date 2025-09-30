#!/usr/bin/env python3
"""
Unified post-processing for NBV, Random, Greedy, Genetic, and SA.
Generates:
- results2/final_2D_table/summary_statistics.csv
- results2/final_2D_table/comparison_all_methods.png (Coverage, Redundancy, Affinity, Normalized Overlap Entropy vs Time)
- results2/final_2D_table/summary_table.tex (LaTeX table with mean ± std)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# Helpers for entropy loading
# ---------------------------
def _reduce_entropy_shape(raw_entropy):
    """
    Accepts:
      - 1D array over time: (T,)
      - 2D array: (T,V) or (V,T) -> mean over viewpoints -> (T,)
      - 3D array: try (T,V,K) or (V,K,T) -> mean over last non-time dims -> (T,)
    Returns 1D array over time or None if unhandled.
    """
    arr = np.asarray(raw_entropy)
    if arr.ndim == 1:
        return arr

    if arr.ndim == 2:
        # choose the axis with larger size as time (fallback heuristic)
        t_axis = 0 if arr.shape[0] >= arr.shape[1] else 1
        return np.nanmean(arr, axis=1-t_axis) if t_axis == 0 else np.nanmean(arr, axis=0)

    if arr.ndim == 3:
        # Try to detect time axis: prefer axis with largest size if monotonic/longest
        t_axis = int(np.argmax(arr.shape))
        # Reduce other axes by mean
        axes = [0,1,2]
        axes.remove(t_axis)
        tmp = np.nanmean(arr, axis=axes[0])
        tmp = np.nanmean(tmp, axis=axes[1]-1)  # after first reduction, axis index shifts by -1
        return tmp

    return None

def _entropy_from_overlap_matrix_series(mat_series):
    """
    Build a time series of mean per-viewpoint raw entropies from an overlap matrix series.
    Accepts arrays shaped (T,V,V) or (V,V,T).
    For each time t and each viewpoint i: take row i (excluding i), normalize to probs, compute H.
    Return array shape (T,).
    """
    A = np.asarray(mat_series, dtype=float)
    if A.ndim != 3:
        return None
    # Arrange as (T,V,V)
    if A.shape[0] != A.shape[1] and A.shape[1] == A.shape[2]:
        # assume (V,V,T) -> transpose to (T,V,V)
        A = np.transpose(A, (2,0,1))
    T, V, V2 = A.shape
    if V != V2:
        return None

    eps = 1e-12
    H_t = np.zeros(T, dtype=float)
    for t in range(T):
        M = A[t]
        # clamp negatives, avoid NaNs
        M = np.clip(M, 0.0, None)
        # force diagonal to zero for overlap distribution over "others"
        np.fill_diagonal(M, 0.0)
        ent = 0.0
        for i in range(V):
            row = M[i]
            s = row.sum()
            if s <= eps:
                # no overlap with others -> entropy 0
                continue
            p = row / (s + eps)
            p_pos = p[p > 0]
            ent += -np.sum(p_pos * np.log(p_pos))
        H_t[t] = ent / max(1, V)  # mean over viewpoints
    return H_t

def _compute_norm_entropy_from_metrics(metrics, num_viewpoints):
    """
    Try multiple ways to produce a normalized overlap entropy series (T,):
      1) Direct raw entropy keys -> reduce -> normalize.
      2) Overlap matrix series -> reconstruct raw -> normalize.
    """
    # 1) Direct raw overlap entropy keys
    possible_keys = [
        'overlap_entropy', 'entropy', 'overlap_entropy_mean', 'entropy_mean',
        'raw_overlap_entropy', 'raw_entropy', 'H_overlap', 'H'
    ]
    for k in possible_keys:
        if k in metrics:
            series = _reduce_entropy_shape(metrics[k])
            if series is not None:
                return _normalize_entropy_series(series, num_viewpoints)

    # 2) Reconstruct from an overlap matrix time-series if available
    matrix_keys = ['overlap_matrix_series', 'overlap_matrices', 'overlap_matrix_t', 'overlap_matrix']
    for k in matrix_keys:
        if k in metrics:
            series = _entropy_from_overlap_matrix_series(metrics[k])
            if series is not None:
                return _normalize_entropy_series(series, num_viewpoints)

    # If nothing worked
    return None

def _normalize_entropy_series(raw_entropy_series, num_viewpoints):
    """Normalized overlap entropy over time: H / ln(N-1), clipped to [0,1]."""
    if raw_entropy_series is None:
        return None
    raw_entropy_series = np.asarray(raw_entropy_series, dtype=float)
    if num_viewpoints is None or num_viewpoints <= 1:
        return np.zeros_like(raw_entropy_series)
    denom = np.log(max(2, num_viewpoints - 1))
    if denom <= 0:
        return np.zeros_like(raw_entropy_series)
    return np.clip(raw_entropy_series / denom, 0.0, 1.0)

# ---------------------------
# Data loading
# ---------------------------
def load_all_data(configs):
    data_by_strategy = {}
    for config in configs:
        strategy = config['strategy']
        file_prefix = config.get('file_prefix', strategy)
        pattern = config.get('pattern', f"{file_prefix}_seed{{}}_metrics.npz")
        base_dir = config['dir']
        data = []
        seed_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("seed")]) if os.path.isdir(base_dir) else []

        for seed_dir in seed_dirs:
            seed_num = int(seed_dir.replace('seed', ''))
            full_path = os.path.join(base_dir, seed_dir, pattern.format(seed_num))
            if os.path.exists(full_path):
                try:
                    metrics = np.load(full_path)
                    # Required series
                    time = metrics['time']
                    coverage = metrics['coverage']
                    redundancy = metrics['redundancy']
                    affinity = metrics['affinity']
                    num_viewpoints = int(metrics['num_viewpoints'])
                    num_iterations = int(metrics['num_iterations'])

                    # Normalized overlap entropy (robust extraction)
                    norm_entropy = _compute_norm_entropy_from_metrics(metrics, num_viewpoints)

                    # Optional exclusive area series if present
                    exclusive_series = None
                    for k in ['exclusive_area', 'exclusive', 'exclusive_percent', 'exclusive_area_percent']:
                        if k in metrics:
                            exclusive_series = np.asarray(metrics[k])
                            # reduce to time series if 2D (T,V) -> mean across viewpoints
                            if exclusive_series.ndim == 2:
                                # prefer (T,V); if (V,T), transpose
                                if exclusive_series.shape[0] < exclusive_series.shape[1]:
                                    exclusive_series = exclusive_series.T
                                exclusive_series = np.nanmean(exclusive_series, axis=1)
                            break

                    data.append({
                        'seed': seed_num,
                        'time': time,
                        'coverage': coverage,
                        'redundancy': redundancy,
                        'affinity': affinity,
                        'norm_entropy': norm_entropy,         # (T,) or None
                        'exclusive_series': exclusive_series, # (T,) or None
                        'num_viewpoints': num_viewpoints,
                        'num_iterations': num_iterations,
                    })
                except Exception as e:
                    print(f"Warning: Could not load {full_path}: {e}")
            else:
                print(f"Warning: Missing {full_path}")
        data_by_strategy[strategy] = data
    return data_by_strategy

# ---------------------------
# Summaries
# ---------------------------
def compute_auc(values, times):
    if values is None or len(values) < 2 or len(times) != len(values):
        return np.nan
    return np.trapz(values, x=times)

def compute_summary(data_by_strategy):
    summaries = []
    for strategy, data_list in data_by_strategy.items():
        if not data_list:
            continue

        def stats(values):
            arr = np.array(values, dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                return np.nan, np.nan
            return float(np.mean(arr)), float(np.std(arr))

        # finals
        final_coverage   = [d['coverage'][-1]   for d in data_list]
        final_redundancy = [d['redundancy'][-1] for d in data_list]
        final_affinity   = [d['affinity'][-1]   for d in data_list]
        final_norm_entropy = [
            (d['norm_entropy'][-1] if d['norm_entropy'] is not None else np.nan)
            for d in data_list
        ]
        final_exclusive = [
            (d['exclusive_series'][-1] if d.get('exclusive_series', None) is not None else np.nan)
            for d in data_list
        ]

        # counts
        num_viewpoints = [d['num_viewpoints'] for d in data_list]
        num_iterations = [d['num_iterations'] for d in data_list]
        sim_time       = [d['time'][-1] for d in data_list]

        # AUCs
        coverage_auc   = [compute_auc(d['coverage'],   d['time']) for d in data_list]
        redundancy_auc = [compute_auc(d['redundancy'], d['time']) for d in data_list]
        affinity_auc   = [compute_auc(d['affinity'],   d['time']) for d in data_list]
        norm_entropy_auc = [
            compute_auc(d['norm_entropy'], d['time']) if d['norm_entropy'] is not None else np.nan
            for d in data_list
        ]

        summary = {
            'strategy': strategy,
            'n_seeds': len(data_list),

            'final_coverage_mean':    stats(final_coverage)[0],
            'final_coverage_std':     stats(final_coverage)[1],

            'final_redundancy_mean':  stats(final_redundancy)[0],
            'final_redundancy_std':   stats(final_redundancy)[1],

            'final_affinity_mean':    stats(final_affinity)[0],
            'final_affinity_std':     stats(final_affinity)[1],

            'final_norm_entropy_mean': stats(final_norm_entropy)[0],
            'final_norm_entropy_std':  stats(final_norm_entropy)[1],

            'final_exclusive_mean':   stats(final_exclusive)[0],
            'final_exclusive_std':    stats(final_exclusive)[1],

            'num_viewpoints_mean':    stats(num_viewpoints)[0],
            'num_viewpoints_std':     stats(num_viewpoints)[1],

            'num_iterations_mean':    stats(num_iterations)[0],
            'num_iterations_std':     stats(num_iterations)[1],

            'simulation_time_mean':   stats(sim_time)[0],
            'simulation_time_std':    stats(sim_time)[1],

            'coverage_auc_mean':      stats(coverage_auc)[0],
            'coverage_auc_std':       stats(coverage_auc)[1],

            'redundancy_auc_mean':    stats(redundancy_auc)[0],
            'redundancy_auc_std':     stats(redundancy_auc)[1],

            'affinity_auc_mean':      stats(affinity_auc)[0],
            'affinity_auc_std':       stats(affinity_auc)[1],

            'norm_entropy_auc_mean':  stats(norm_entropy_auc)[0],
            'norm_entropy_auc_std':   stats(norm_entropy_auc)[1],
        }
        summaries.append(summary)
    return pd.DataFrame(summaries)

# ---------------------------
# Plotting
# ---------------------------
def _interp_stack(data, key, common_time):
    """Interpolate a list of time series to a common grid; skip None."""
    series = []
    for d in data:
        y = d[key]
        if y is None:
            continue
        series.append(np.interp(common_time, d['time'], y))
    if not series:
        return None
    return np.vstack(series)

def plot_comparative_curves(data_by_strategy, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    colors = {
        'nbv': 'blue', 'random': 'red', 'greedy': 'purple',
        'genetic': 'green', 'sa': 'orange'
    }

    # Build a global common time grid based on max end time across all strategies/seeds
    max_t = 0.0
    for data in data_by_strategy.values():
        if not data:
            continue
        mt = max((d['time'][-1] for d in data if len(d['time']) > 0), default=0.0)
        max_t = max(max_t, mt)
    common_time = np.linspace(0, max_t, 500) if max_t > 0 else np.linspace(0, 1, 2)

    for strategy, data in data_by_strategy.items():
        if not data:
            continue
        color = colors.get(strategy, 'gray')
        label = strategy.upper()

        for ax_idx, metric_key in [((0,0), 'coverage'), ((0,1), 'redundancy'), ((1,0), 'affinity')]:
            stack = _interp_stack(data, metric_key, common_time)
            if stack is None: continue
            mean, std = np.nanmean(stack, axis=0), np.nanstd(stack, axis=0)
            axes[ax_idx].plot(common_time, mean, label=label, color=color)
            axes[ax_idx].fill_between(common_time, mean - std, mean + std, alpha=0.25, color=color)

        stack_ne = _interp_stack(data, 'norm_entropy', common_time)
        if stack_ne is not None:
            mean_ne, std_ne = np.nanmean(stack_ne, axis=0), np.nanstd(stack_ne, axis=0)
            axes[1,1].plot(common_time, mean_ne, label=label, color=color)
            axes[1,1].fill_between(common_time, mean_ne - std_ne, mean_ne + std_ne, alpha=0.25, color=color)

    axes[0,0].set_title('Coverage Over Time')
    axes[0,1].set_title('Redundancy Over Time')
    axes[1,0].set_title('Affinity Over Time')
    axes[1,1].set_title('Normalized Overlap Entropy Over Time')

    for ax in axes.flat:
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[1,1].set_ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot: {save_path}")

# ---------------------------
# LaTeX table writer
# ---------------------------
def _fmt(mean, std, digits=2):
    if pd.isna(mean) or pd.isna(std):
        return r"--"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"

def write_latex_table(summary_df, out_dir):
    cols = set(summary_df.columns)

    # Column names we expect
    vp_mean, vp_std = "num_viewpoints_mean", "num_viewpoints_std"
    t_mean,  t_std  = "simulation_time_mean", "simulation_time_std"
    ex_mean, ex_std = "final_exclusive_mean", "final_exclusive_std"
    ne_mean, ne_std = "final_norm_entropy_mean", "final_norm_entropy_std"

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Summary across strategies: mean $\pm$ std.}")
    lines.append(r"\label{tab:summary_strategies}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Strategy & \# Viewpoints & Solution Time (s) & Exclusive Area (\%) & Overlap Entropy (normalized) \\")
    lines.append(r"\midrule")

    for _, r in summary_df.iterrows():
        strategy = str(r.get("strategy", "")).upper()
        vp = _fmt(r.get(vp_mean, np.nan), r.get(vp_std, np.nan), digits=2)
        tm = _fmt(r.get(t_mean,  np.nan), r.get(t_std,  np.nan), digits=2)
        ex = _fmt(r.get(ex_mean, np.nan), r.get(ex_std, np.nan), digits=2) if (ex_mean in cols and ex_std in cols) else r"--"
        ne = _fmt(r.get(ne_mean, np.nan), r.get(ne_std, np.nan), digits=3) if (ne_mean in cols and ne_std in cols) else r"--"
        lines.append(f"{strategy} & {vp} & {tm} & {ex} & {ne} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    tex_path = os.path.join(out_dir, "summary_table.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"Wrote LaTeX table: {tex_path}")

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='./results2')
    parser.add_argument('--k_attr', type=float, default=10.0)
    parser.add_argument('--k_rep', type=float, default=0.25)
    # Default output dir: results2/final_2D_table
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    default_out = os.path.join(args.results_dir, 'final_2D_table')
    output_dir = args.output_dir or default_out
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

    # Write LaTeX table with the requested columns
    write_latex_table(summary_df, output_dir)

    print("Done.")

if __name__ == '__main__':
    main()
