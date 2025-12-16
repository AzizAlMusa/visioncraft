#!/usr/bin/env python
"""
analyze_viewset_rl_training.py

Utility script to analyze training logs produced by
train_viewset_rl_single_model.py and generate summary statistics
and plots.

Expected input file:
  training_metrics_viewset_rl.npz

Keys inside the .npz:
  - rewards         : (T,)
  - coverages       : (T,)
  - losses          : (T,)
  - baselines       : (T,)
  - num_views_used  : (T,)
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def moving_average(x, window):
    """
    Simple moving average with 'valid' style for the interior and
    edge handling by shrinking the effective window near boundaries.
    """
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if n == 0:
        return x
    if window <= 1:
        return x.copy()

    half = window // 2
    out = np.zeros_like(x, dtype=np.float32)
    for i in range(n):
        left = max(0, i - half)
        right = min(n, i + half + 1)
        out[i] = x[left:right].mean()
    return out


def summarize_metrics(rewards, coverages, losses, baselines, num_views_used, window=50):
    """
    Print a small textual summary of training behavior.
    """
    T = len(rewards)
    episodes = np.arange(1, T + 1)

    best_cov_idx = int(np.argmax(coverages))
    best_cov = float(coverages[best_cov_idx])
    best_cov_ep = int(episodes[best_cov_idx])

    best_rew_idx = int(np.argmax(rewards))
    best_rew = float(rewards[best_rew_idx])
    best_rew_ep = int(episodes[best_rew_idx])

    last_k = min(window, T)

    mean_cov_last = float(np.mean(coverages[-last_k:]))
    mean_rew_last = float(np.mean(rewards[-last_k:]))
    mean_loss_last = float(np.mean(losses[-last_k:]))
    mean_views_last = float(np.mean(num_views_used[-last_k:]))

    mean_views = float(np.mean(num_views_used))
    min_views = int(np.min(num_views_used))
    max_views = int(np.max(num_views_used))

    print("===== Viewset RL Training Summary =====")
    print(f"Total episodes           : {T}")
    print("")
    print(f"Best coverage            : {best_cov:.4f} at episode {best_cov_ep}")
    print(f"Best reward              : {best_rew:.4f} at episode {best_rew_ep}")
    print("")
    print(f"Final coverage           : {float(coverages[-1]):.4f}")
    print(f"Final reward             : {float(rewards[-1]):.4f}")
    print(f"Final loss               : {float(losses[-1]):.4f}")
    print(f"Final baseline           : {float(baselines[-1]):.4f}")
    print("")
    print(f"Mean coverage (last {last_k:3d}) : {mean_cov_last:.4f}")
    print(f"Mean reward   (last {last_k:3d}) : {mean_rew_last:.4f}")
    print(f"Mean loss     (last {last_k:3d}) : {mean_loss_last:.4f}")
    print(f"Mean views    (last {last_k:3d}) : {mean_views_last:.2f}")
    print("")
    print("Views used per episode:")
    print(f"  mean                     : {mean_views:.2f}")
    print(f"  min                      : {min_views}")
    print(f"  max                      : {max_views}")
    print("=======================================\n")


def plot_reward_and_coverage(episodes, rewards, coverages, out_path, window=50):
    """
    Plot reward and coverage over training with a moving average.
    """
    ma_rewards = moving_average(rewards, window)
    ma_coverages = moving_average(coverages, window)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Reward subplot
    ax = axes[0]
    ax.plot(episodes, rewards, alpha=0.3, label="Reward (raw)")
    ax.plot(episodes, ma_rewards, linewidth=2.0, label=f"Reward (MA, window={window})")
    ax.set_ylabel("Reward")
    ax.set_title("Reward over episodes")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")

    # Coverage subplot
    ax = axes[1]
    ax.plot(episodes, coverages, alpha=0.3, label="Coverage (raw)")
    ax.plot(episodes, ma_coverages, linewidth=2.0, label=f"Coverage (MA, window={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Coverage")
    ax.set_title("Coverage over episodes")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[SAVE] Reward/Coverage plot -> {out_path}")


def plot_loss_and_baseline(episodes, losses, baselines, out_path, window=50):
    """
    Plot loss and baseline (value estimate) over training.
    """
    ma_losses = moving_average(losses, window)
    ma_baselines = moving_average(baselines, window)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    ax.plot(episodes, losses, alpha=0.3, label="Loss (raw)")
    ax.plot(episodes, ma_losses, linewidth=2.0, label=f"Loss (MA, window={window})")
    ax.set_ylabel("Loss")
    ax.set_title("Policy loss over episodes")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(episodes, baselines, alpha=0.3, label="Baseline (raw)")
    ax.plot(episodes, ma_baselines, linewidth=2.0, label=f"Baseline (MA, window={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Baseline")
    ax.set_title("Baseline (running reward estimate)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[SAVE] Loss/Baseline plot -> {out_path}")


def plot_views_used(episodes, num_views_used, out_path, window=50):
    """
    Plot how many views are used per episode (time series + histogram).
    """
    ma_views = moving_average(num_views_used, window)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    ax = axes[0]
    ax.plot(episodes, num_views_used, alpha=0.3, label="Views used (raw)")
    ax.plot(episodes, ma_views, linewidth=2.0, label=f"Views used (MA, window={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("# views")
    ax.set_title("Number of selected views per episode")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")

    ax = axes[1]
    ax.hist(num_views_used, bins=np.arange(num_views_used.min(), num_views_used.max() + 2) - 0.5)
    ax.set_xlabel("# views")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of views used")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[SAVE] Views-used plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze training metrics from train_viewset_rl_single_model.py"
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default="outputs/viewset_rl_single_model/training_metrics_viewset_rl.npz",
        help="Path to training_metrics_viewset_rl.npz"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: same as metrics file directory)"
    )
    parser.add_argument(
        "--ma-window",
        type=int,
        default=50,
        help="Moving average window size for smoothing plots"
    )

    args = parser.parse_args()

    metrics_path = args.metrics_path
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    if args.out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(metrics_path))
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    data = np.load(metrics_path)

    # Required keys as saved in train_viewset_rl_single_model.py
    # rewards, coverages, losses, baselines, num_views_used
    try:
        rewards = data["rewards"]
        coverages = data["coverages"]
        losses = data["losses"]
        baselines = data["baselines"]
        num_views_used = data["num_views_used"]
    except KeyError as e:
        raise KeyError(f"Missing expected key in metrics file: {e}")

    T = len(rewards)
    if not (len(coverages) == T and len(losses) == T and len(baselines) == T and len(num_views_used) == T):
        raise ValueError("All metric arrays must have the same length")

    episodes = np.arange(1, T + 1)

    # Text summary
    summarize_metrics(rewards, coverages, losses, baselines, num_views_used, window=args.ma_window)

    # Plots
    reward_cov_path = os.path.join(out_dir, "viewset_rl_reward_coverage.png")
    loss_baseline_path = os.path.join(out_dir, "viewset_rl_loss_baseline.png")
    views_used_path = os.path.join(out_dir, "viewset_rl_views_used.png")

    plot_reward_and_coverage(
        episodes,
        rewards,
        coverages,
        reward_cov_path,
        window=args.ma_window,
    )
    plot_loss_and_baseline(
        episodes,
        losses,
        baselines,
        loss_baseline_path,
        window=args.ma_window,
    )
    plot_views_used(
        episodes,
        num_views_used,
        views_used_path,
        window=args.ma_window,
    )

    print("Analysis complete.")


if __name__ == "__main__":
    main()
