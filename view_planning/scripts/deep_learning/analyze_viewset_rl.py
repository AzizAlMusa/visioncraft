#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def moving_avg(x, k=50):
    if len(x) < k:
        return x
    c = np.convolve(x, np.ones(k)/k, mode="valid")
    # pad to same length for nicer plotting
    pad = np.full(k-1, np.nan)
    return np.concatenate([pad, c])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=str,
        default="outputs/viewset_rl_single_model/training_metrics_viewset_rl.npz",
        help="Path to npz saved by train_viewset_rl_single_model.py",
    )
    parser.add_argument("--window", type=int, default=50,
                        help="Moving-average window for smoothing")
    args = parser.parse_args()

    if not os.path.exists(args.metrics):
        raise FileNotFoundError(f"Metrics file not found: {args.metrics}")

    data = np.load(args.metrics)
    print("Available keys in metrics file:", data.files)

    episodes = data["episodes"] if "episodes" in data else np.arange(len(data["coverage"]))
    coverage = data["coverage"]
    reward   = data["reward"] if "reward" in data else coverage
    loss     = data["loss"] if "loss" in data else None
    baseline = data["baseline"] if "baseline" in data else None

    print(f"#episodes: {len(episodes)}")
    print(f"coverage: mean={coverage.mean():.4f}, min={coverage.min():.4f}, max={coverage.max():.4f}")
    print(f"reward:   mean={reward.mean():.4f}, min={reward.min():.4f}, max={reward.max():.4f}")

    # --- Plot coverage & reward ---
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, coverage, alpha=0.3, label="coverage (raw)")
    plt.plot(episodes, moving_avg(coverage, args.window), label=f"coverage (MA {args.window})")
    plt.xlabel("Episode")
    plt.ylabel("Coverage")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.title("Coverage vs Episode (viewset RL)")
    plt.tight_layout()

    # --- Reward / baseline ---
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, reward, alpha=0.3, label="reward (raw)")
    plt.plot(episodes, moving_avg(reward, args.window), label=f"reward (MA {args.window})")
    if baseline is not None:
        plt.plot(episodes, baseline, label="baseline (running mean)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.title("Reward / baseline vs Episode")
    plt.tight_layout()

    # --- Loss (optional) ---
    if loss is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, loss, alpha=0.4)
        plt.xlabel("Episode")
        plt.ylabel("Policy loss")
        plt.grid(True)
        plt.title("Policy loss vs Episode")
        plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
