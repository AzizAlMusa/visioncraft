# train_nbv_ppo_single_model.py

import os
import sys
import argparse
import numpy as np
import torch

from nbv_env_discrete import build_env_from_visioncraft
from ppo_discrete import PPOAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings-path", type=str, required=True,
                        help="Path to directory containing visioncraft_py bindings")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to CAD model (.ply)")
    parser.add_argument("--num-candidates", type=int, default=64,
                        help="Number of candidate viewpoints on the sphere")
    parser.add_argument("--max-steps", type=int, default=6,
                        help="Max number of views per episode")
    parser.add_argument("--total-steps", type=int, default=20000,
                        help="Total environment steps to collect for training")
    parser.add_argument("--steps-per-update", type=int, default=1024,
                        help="Steps per PPO update")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--clip-ratio", type=float, default=0.2,
                        help="PPO clip ratio")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="Value loss coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--fixed-radius", type=float, default=400.0,
                        help="Radius of camera sphere")
    parser.add_argument("--downsample", type=float, default=4.0,
                        help="Viewpoint downsample factor for precomputation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str,
                        default="outputs/nbv_ppo_single_model",
                        help="Directory to save checkpoints and metrics")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    # Build env once (this calls Visioncraft to precompute vis_matrix)
    env = build_env_from_visioncraft(
        bindings_path=args.bindings_path,
        model_path=args.model,
        num_candidates=args.num_candidates,
        max_steps=args.max_steps,
        fixed_radius=args.fixed_radius,
        downsample_factor=args.downsample,
        seed=args.seed,
        verbose=True,
    )

    obs_dim = env.obs_dim
    act_dim = env.num_views

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[RL] Using device: {device}")

    agent = PPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        device=device,
    )

    total_steps = 0
    all_ep_returns = []
    all_ep_lengths = []
    all_det_cov = []

    update_idx = 0

    while total_steps < args.total_steps:
        update_idx += 1
        steps_to_collect = min(args.steps_per_update,
                               args.total_steps - total_steps)

        data = agent.collect_rollouts(env, min_steps=steps_to_collect)
        total_steps += len(data["obs"])

        # PPO update
        stats = agent.update(data, train_epochs=10, batch_size=64)

        ep_ret = data["ep_returns"]
        ep_len = data["ep_lengths"]

        all_ep_returns.extend(ep_ret.tolist())
        all_ep_lengths.extend(ep_len.tolist())

        avg_ret = float(ep_ret.mean()) if len(ep_ret) > 0 else 0.0
        avg_len = float(ep_len.mean()) if len(ep_len) > 0 else 0.0

        # Deterministic evaluation: always pick argmax(logits)
        det_cov = evaluate_deterministic(agent, env)
        all_det_cov.append(det_cov)

        print(
            f"[Update {update_idx:03d}] "
            f"steps={total_steps:06d} "
            f"avg_return={avg_ret:.4f} "
            f"avg_len={avg_len:.2f} "
            f"det_cov={det_cov:.4f} "
            f"pi_loss={stats['policy_loss']:.4f} "
            f"v_loss={stats['value_loss']:.4f} "
            f"entropy={stats['entropy']:.4f}"
        )

        # Save checkpoint periodically
        if update_idx % 10 == 0 or total_steps >= args.total_steps:
            ckpt_path = os.path.join(args.out_dir,
                                     f"nbv_ppo_ep_{total_steps:06d}.pt")
            torch.save(
                {
                    "net_state": agent.net.state_dict(),
                    "obs_dim": obs_dim,
                    "act_dim": act_dim,
                },
                ckpt_path,
            )
            print(f"[SAVE] Checkpoint -> {ckpt_path}")

            metrics_path = os.path.join(args.out_dir, "training_metrics_ppo.npz")
            np.savez_compressed(
                metrics_path,
                ep_returns=np.asarray(all_ep_returns, dtype=np.float32),
                ep_lengths=np.asarray(all_ep_lengths, dtype=np.float32),
                det_cov=np.asarray(all_det_cov, dtype=np.float32),
            )
            print(f"[SAVE] Metrics -> {metrics_path}")

    print("Training finished.")
    if len(all_det_cov) > 0:
        print(f"Final deterministic coverage: {all_det_cov[-1]:.4f}")


def evaluate_deterministic(agent: PPOAgent, env) -> float:
    """
    One deterministic evaluation: always pick argmax(logits).
    Returns final coverage of that episode.
    """
    obs = env.reset()
    done = False
    coverage = 0.0
    while not done:
        obs_t = torch.from_numpy(obs).float().to(agent.device)
        with torch.no_grad():
            logits, _ = agent.net(obs_t.unsqueeze(0))
            action = torch.argmax(logits, dim=-1).item()
        obs, _, done, info = env.step(action)
        coverage = info["coverage"]
    return float(coverage)


if __name__ == "__main__":
    main()
