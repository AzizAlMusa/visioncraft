import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def moving_avg(x, k=50):
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k)/k, mode='valid')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=str, required=True,
                        help="Path to training_metrics_viewset_rl.npz")
    parser.add_argument("--out", type=str, default="analysis_plots",
                        help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    data = np.load(args.metrics)
    rewards = data["rewards"]
    coverages = data["coverages"]
    losses = data["losses"]
    baselines = data["baselines"]

    ma_cov = moving_avg(coverages)
    ma_rew = moving_avg(rewards)

    # ------------------------------
    # 1. Coverage curve
    # ------------------------------
    plt.figure(figsize=(8,4))
    plt.plot(coverages, alpha=0.3, label="coverage")
    plt.plot(ma_cov, linewidth=2, label="moving avg (50)")
    plt.xlabel("episode")
    plt.ylabel("coverage")
    plt.title("Coverage progression")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(args.out, "coverage.png"), dpi=200)

    # ------------------------------
    # 2. Reward curve
    # ------------------------------
    plt.figure(figsize=(8,4))
    plt.plot(rewards, alpha=0.3, label="reward")
    plt.plot(ma_rew, linewidth=2, label="moving avg (50)")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.title("Reward progression")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(args.out, "reward.png"), dpi=200)

    # ------------------------------
    # 3. Loss
    # ------------------------------
    plt.figure(figsize=(8,4))
    plt.plot(losses, label="loss")
    plt.xlabel("episode")
    plt.ylabel("loss")
    plt.title("Policy loss")
    plt.grid(True)
    plt.savefig(os.path.join(args.out, "loss.png"), dpi=200)

    # ------------------------------
    # 4. Baseline vs reward
    # ------------------------------
    plt.figure(figsize=(8,4))
    plt.plot(baselines, label="baseline")
    plt.plot(rewards, alpha=0.3, label="reward")
    plt.xlabel("episode")
    plt.ylabel("value")
    plt.title("Baseline vs reward")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(args.out, "baseline_comparison.png"), dpi=200)

    # ------------------------------
    # Additional metrics
    # ------------------------------
    best_cov = float(np.max(coverages))
    early = np.argmax(coverages >= 0.90) if np.any(coverages >= 0.90) else None
    mid = np.argmax(coverages >= 0.95) if np.any(coverages >= 0.95) else None
    high = np.argmax(coverages >= 0.99) if np.any(coverages >= 0.99) else None

    last_var = float(np.var(coverages[-200:]))

    print("\n===== Analysis =====")
    print("Best coverage:", best_cov)
    print("Episode reach 0.90:", early)
    print("Episode reach 0.95:", mid)
    print("Episode reach 0.99:", high)
    print("Coverage variance (last 200 episodes):", last_var)
    print("Plots saved to:", args.out)

if __name__ == "__main__":
    main()
