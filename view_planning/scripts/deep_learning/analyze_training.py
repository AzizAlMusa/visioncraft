#!/usr/bin/env python3
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=str,
        default="outputs/viewformer_single_model/training_metrics.npz",
        help="Path to training_metrics.npz",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/viewformer_single_model/analysis",
        help="Directory to save plots",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.metrics):
        print(f"[ERROR] metrics file not found: {args.metrics}")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    data = np.load(args.metrics)
    episode = data["episode"]
    reward = data["reward"]
    coverage = data["coverage"]
    num_views_used = data["num_views_used"]
    avg_gate = data["avg_gate"]

    print(f"[INFO] Loaded metrics for {len(episode)} episodes")
    print(f"  reward:   mean={reward.mean():.4f}, min={reward.min():.4f}, max={reward.max():.4f}")
    print(f"  coverage: mean={coverage.mean():.4f}, min={coverage.min():.4f}, max={coverage.max():.4f}")
    print(f"  views:    mean={num_views_used.mean():.2f}")

    # 1) reward vs episode
    plt.figure()
    plt.plot(episode, reward, linewidth=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward vs Episode")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "reward_vs_episode.png"), dpi=150)

    # 2) coverage vs episode
    plt.figure()
    plt.plot(episode, coverage, linewidth=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Coverage")
    plt.title("Coverage vs Episode")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "coverage_vs_episode.png"), dpi=150)

    # 3) views vs episode
    plt.figure()
    plt.plot(episode, num_views_used, linewidth=0.7)
    plt.xlabel("Episode")
    plt.ylabel("# views used")
    plt.title("Views used vs Episode")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "views_vs_episode.png"), dpi=150)

    # 4) coverage vs views
    plt.figure()
    plt.scatter(num_views_used, coverage, s=5, alpha=0.5)
    plt.xlabel("# views used")
    plt.ylabel("Coverage")
    plt.title("Coverage vs # views")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "coverage_vs_views.png"), dpi=150)

    print(f"[SAVE] Plots written to {args.out_dir}")


if __name__ == "__main__":
    main()
