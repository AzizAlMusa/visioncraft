# eval_nbv_ppo_single_model.py

import os
import sys
import argparse
import numpy as np
import torch

from nbv_env_discrete import build_env_from_visioncraft
from ppo_discrete import PolicyValueNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings-path", type=str, required=True,
                        help="Path to directory containing visioncraft_py bindings")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to CAD model (.ply)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to PPO checkpoint .pt")
    parser.add_argument("--num-candidates", type=int, default=64,
                        help="Number of candidate viewpoints (must match training)")
    parser.add_argument("--max-steps", type=int, default=6,
                        help="Max views per episode (must match training)")
    parser.add_argument("--fixed-radius", type=float, default=400.0,
                        help="Camera sphere radius (must match training)")
    parser.add_argument("--downsample", type=float, default=4.0,
                        help="Downsample factor (must match training)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of eval episodes")
    args = parser.parse_args()

    env = build_env_from_visioncraft(
        bindings_path=args.bindings_path,
        model_path=args.model,
        num_candidates=args.num_candidates,
        max_steps=args.max_steps,
        fixed_radius=args.fixed_radius,
        downsample_factor=args.downsample,
        seed=0,
        verbose=False,
    )

    obs_dim = env.obs_dim
    act_dim = env.num_views

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[EVAL] Using device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    net = PolicyValueNet(obs_dim, act_dim).to(device)
    net.load_state_dict(ckpt["net_state"])
    net.eval()

    covs = []
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        final_cov = 0.0
        actions = []

        while not done:
            obs_t = torch.from_numpy(obs).float().to(device)
            with torch.no_grad():
                logits, _ = net(obs_t.unsqueeze(0))
                action = torch.argmax(logits, dim=-1).item()

            actions.append(action)
            obs, _, done, info = env.step(action)
            final_cov = info["coverage"]

        covs.append(final_cov)
        print(f"[Ep {ep+1:03d}] coverage={final_cov:.4f}, actions={actions}")

    covs = np.array(covs, dtype=np.float32)
    print(f"[EVAL] mean coverage = {covs.mean():.4f}, "
          f"max = {covs.max():.4f}, min = {covs.min():.4f}")


if __name__ == "__main__":
    main()
