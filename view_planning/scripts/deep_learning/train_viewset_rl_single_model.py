# train_viewset_rl_single_model.py
#
# Actor critic style training on a single model:
# - One shot viewset on a sphere
# - Gating over a maximum number of candidate viewpoints
# - Annealed Gaussian exploration in direction space
# - Annealed gating temperature
# - Optional diversity regularizer on mean directions

import os
import sys
import time
import argparse
import numpy as np
import torch

from viewset_policy import ViewSetPolicy


def viewpoint_diversity_loss(mean_dirs: torch.Tensor) -> torch.Tensor:
    """
    Simple diversity regularizer on the mean directions.

    Encourages candidate view directions not to collapse to the same vector.

    mean_dirs: (K, 3) tensor of unit vectors.
    Returns a scalar tensor.
    """
    if mean_dirs.ndim != 2 or mean_dirs.shape[0] < 2:
        return mean_dirs.new_tensor(0.0)

    # Re normalize just in case
    dirs = mean_dirs / (mean_dirs.norm(dim=-1, keepdim=True) + 1e-8)  # (K,3)
    K = dirs.shape[0]

    # Cosine similarity matrix
    cos = dirs @ dirs.t()  # (K,K)

    # Remove diagonal (self similarity)
    mask = torch.ones_like(cos, dtype=torch.bool)
    idx = torch.arange(K, device=dirs.device)
    mask[idx, idx] = False
    off_diag = cos[mask]

    # Penalize large squared similarity
    return (off_diag ** 2).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings-path", type=str, default="../../build/python_bindings",
                        help="Path to directory containing visioncraft_py module")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to .ply model")
    parser.add_argument("--episodes", type=int, default=2000,
                        help="Number of RL episodes")
    parser.add_argument("--num-views", type=int, default=8,
                        help="Maximum number of candidate viewpoints per episode (K_max)")
    parser.add_argument("--pc-size", type=int, default=2048,
                        help="Number of points used as input")
    parser.add_argument("--radius", type=float, default=400.0,
                        help="Sphere radius for viewpoints (world units)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--sigma-start", type=float, default=0.3,
                        help="Initial stddev of Gaussian exploration in direction space")
    parser.add_argument("--sigma-end", type=float, default=0.05,
                        help="Final stddev of Gaussian exploration in direction space")
    parser.add_argument("--gate-temp-start", type=float, default=1.5,
                        help="Initial temperature for gating logits (higher means smoother probs)")
    parser.add_argument("--gate-temp-end", type=float, default=0.5,
                        help="Final temperature for gating logits (lower means sharper probs)")
    parser.add_argument("--entropy-coef", type=float, default=1e-3,
                        help="Entropy regularization coefficient")
    parser.add_argument("--value-coef", type=float, default=0.5,
                        help="Coefficient for value loss (critic)")
    parser.add_argument("--dir-div-coef", type=float, default=0.01,
                        help="Coefficient for viewpoint diversity regularizer")
    parser.add_argument("--view-cost", type=float, default=0.05,
                        help="Cost per selected view; larger values push the policy to use fewer views")
    parser.add_argument("--min-views", type=int, default=1,
                        help="Minimum number of views that must be active per episode")
    parser.add_argument("--out-dir", type=str, default="outputs/viewset_rl_single_model",
                        help="Output directory")
    args = parser.parse_args()

    # Setup paths and device
    os.makedirs(args.out_dir, exist_ok=True)
    sys.path.append(args.bindings_path)

    from visioncraft_py import Model, Viewpoint, VisibilityManager  # noqa: E402
    from viewset_env import ViewSetEnv  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Build environment
    env = ViewSetEnv(
        model_path=args.model,
        Model=Model,
        Viewpoint=Viewpoint,
        VisibilityManager=VisibilityManager,
        num_views=args.num_views,   # interpreted as a maximum; eval is variable K
        radius=args.radius,
        pc_size=args.pc_size,
        device=device,
        debug=True,   # prints once at init and first eval
    )

    pc = env.get_state()  # (pc_size, 3) tensor on device

    # Policy with built in value head
    net = ViewSetPolicy(num_views=args.num_views).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # Simple running mean baseline for logging
    baseline = 0.0

    rewards = []
    coverages = []
    losses = []
    baselines = []
    used_views = []

    start_time = time.time()

    def interp(start, end, frac):
        return start + frac * (end - start)

    for ep in range(1, args.episodes + 1):
        t0 = time.time()
        frac = 0.0 if args.episodes <= 1 else float(ep - 1) / float(args.episodes - 1)

        # Annealed exploration stddev and gating temperature
        sigma_ep = float(interp(args.sigma_start, args.sigma_end, frac))
        gate_temp_ep = float(interp(args.gate_temp_start, args.gate_temp_end, frac))
        sigma_sq = sigma_ep ** 2
        log_sigma = np.log(sigma_ep)

        # Forward pass: mean directions, gating logits, and state value
        net.train()
        mean_dirs, gate_logits, value_pred = net(pc, return_value=True)  # (K_max,3), (K_max,), scalar

        # Sample Gaussian noise in R^3 per candidate view
        eps = torch.randn_like(mean_dirs)
        sampled_dirs = mean_dirs + sigma_ep * eps
        # Renormalize to unit sphere
        sampled_dirs = sampled_dirs / (sampled_dirs.norm(dim=-1, keepdim=True) + 1e-8)

        # Bernoulli gating over candidates with annealed temperature
        gate_probs = torch.sigmoid(gate_logits / gate_temp_ep)  # (K_max,)
        bernoulli = torch.distributions.Bernoulli(probs=gate_probs)
        gate_sample = bernoulli.sample()                         # (K_max,), values in {0,1}

        # Ensure at least min views are active
        num_on = int(gate_sample.sum().item())
        if num_on < args.min_views:
            k = min(args.min_views, gate_probs.shape[0])
            # Pick top k most likely views
            _, top_idx = torch.topk(gate_probs, k=k)
            gate_sample = torch.zeros_like(gate_sample)
            gate_sample[top_idx] = 1.0
            num_on = k

        # Mask directions to only selected ones
        mask = gate_sample.bool()
        selected_dirs = sampled_dirs[mask]  # (M,3)

        # Evaluate coverage with fresh VisibilityManager
        coverage = env.evaluate_viewset(selected_dirs, debug=(ep == 1 or ep % 200 == 0))

        # Shaped coverage reward
        tau = 0.97
        lam = 5.0
        bonus = max(0.0, (coverage - tau) / (1.0 - tau))  # normalized in [0, 1]

        # Penalize number of views used
        reward = coverage + lam * bonus - args.view_cost * float(num_on)

        # Log probability under
        #  - Normal(mean_dirs, sigma^2 I) for directions
        #  - Bernoulli(gate_probs) for gating
        diff = (sampled_dirs - mean_dirs)
        log_prob_per_dim = -0.5 * (diff ** 2 / sigma_sq + 2.0 * log_sigma + np.log(2.0 * np.pi))
        log_prob_dirs = log_prob_per_dim.sum()  # sum over all views and dims

        log_prob_gate = bernoulli.log_prob(gate_sample).sum()
        log_prob = log_prob_dirs + log_prob_gate

        # Entropy
        entropy_dirs_per_dim = 0.5 * (1.0 + np.log(2.0 * np.pi * sigma_sq))
        entropy_dirs = entropy_dirs_per_dim * 3.0 * args.num_views   # scalar
        entropy_gate = bernoulli.entropy().sum()
        entropy_total = entropy_dirs + entropy_gate

        # Running mean baseline, for logging only
        if ep == 1:
            baseline = reward
        else:
            baseline = 0.99 * baseline + 0.01 * reward

        # Critic
        value_pred_scalar = value_pred.squeeze()
        value_target = torch.tensor(reward, dtype=torch.float32, device=device)
        advantage_tensor = value_target - value_pred_scalar.detach()
        advantage = float(advantage_tensor.item())

        policy_loss = -log_prob * advantage_tensor
        value_loss = 0.5 * (value_pred_scalar - value_target) ** 2
        entropy_loss = -args.entropy_coef * entropy_total

        # Diversity regularizer on mean directions
        div_loss = viewpoint_diversity_loss(mean_dirs)

        loss = policy_loss + args.value_coef * value_loss + entropy_loss + args.dir_div_coef * div_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rewards.append(reward)
        coverages.append(coverage)
        losses.append(float(loss.detach().cpu().item()))
        baselines.append(baseline)
        used_views.append(num_on)

        elapsed_ep = time.time() - t0
        recent_mean_cov = np.mean(coverages[-50:]) if len(coverages) >= 50 else np.mean(coverages)
        recent_mean_views = np.mean(used_views[-50:]) if len(used_views) >= 50 else np.mean(used_views)

        print(
            "[Ep {:04d}] reward={:.4f} cov={:.4f} views={:2d} adv={:.4f} loss={:.4f} "
            "baseline={:.4f} recent_cov={:.4f} recent_views={:.2f} sigma={:.3f} gate_temp={:.3f} elapsed={:.1f}s".format(
                ep,
                reward,
                coverage,
                num_on,
                advantage,
                float(loss.detach().cpu().item()),
                baseline,
                recent_mean_cov,
                recent_mean_views,
                sigma_ep,
                gate_temp_ep,
                elapsed_ep,
            )
        )

    total_time = time.time() - start_time
    print("Training finished in {:.1f}s.".format(total_time))
    print("Best coverage seen:", np.max(coverages))

    # Save checkpoint and metrics
    ckpt_path = os.path.join(
        args.out_dir,
        "viewset_rl_ep_{:06d}.pt".format(args.episodes),
    )
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "config": vars(args),
            "baseline": baseline,
        },
        ckpt_path,
    )
    print("[SAVE] Checkpoint ->", ckpt_path)

    metrics_path = os.path.join(args.out_dir, "training_metrics_viewset_rl.npz")
    np.savez_compressed(
        metrics_path,
        rewards=np.array(rewards, dtype=np.float32),
        coverages=np.array(coverages, dtype=np.float32),
        losses=np.array(losses, dtype=np.float32),
        baselines=np.array(baselines, dtype=np.float32),
        num_views_used=np.array(used_views, dtype=np.int32),
    )
    print("[SAVE] Metrics ->", metrics_path)


if __name__ == "__main__":
    main()
