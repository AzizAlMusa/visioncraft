#!/usr/bin/env python3
"""
eval_viewset_rl_single_model.py

Load a trained continuous view-set policy checkpoint and:

- run several deterministic evaluations (using the mean angles),
- report coverage statistics,
- optionally visualize the viewpoints + visibility in Visioncraft.

Usage:

    python eval_viewset_rl_single_model.py \
        --bindings-path ../../build/python_bindings \
        --model ../../models/gorilla.ply \
        --checkpoint outputs/viewset_rl_single_model/viewset_rl_ep_002000.pt \
        --num-views 6 \
        --pc-size 2048 \
        --radius 400.0 \
        --visualize
"""

import os
import sys
import math
import argparse
import numpy as np
import torch

from viewset_env_single_model import ViewSetEnvConfig, ViewSetEnvSingleModel
from viewset_policy import ViewSetPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings-path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-views", type=int, default=6)
    parser.add_argument("--pc-size", type=int, default=2048)
    parser.add_argument("--radius", type=float, default=400.0)
    parser.add_argument("--near", type=float, default=300.0)
    parser.add_argument("--far", type=float, default=900.0)
    parser.add_argument("--downsample", type=float, default=2.0)
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of deterministic eval episodes")
    parser.add_argument("--visualize", action="store_true",
                        help="Open Visioncraft Visualizer with last view-set")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Build env
    # ------------------------------------------------------------------ #
    cfg = ViewSetEnvConfig(
        bindings_path=args.bindings_path,
        model_path=args.model,
        num_views=args.num_views,
        radius=args.radius,
        pc_size=args.pc_size,
        near=args.near,
        far=args.far,
        downsample_factor=args.downsample,
        device="cuda",
    )
    env = ViewSetEnvSingleModel(cfg)
    device = env.device
    print(f"\n[EVAL] Using device: {device}")

    # ------------------------------------------------------------------ #
    # Load policy
    # ------------------------------------------------------------------ #
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "policy" not in ckpt:
        raise KeyError(
            f"Checkpoint {args.checkpoint} does not contain key 'policy'. "
            f"Keys available: {list(ckpt.keys())}"
        )

    policy = ViewSetPolicy(num_views=args.num_views, sigma_init=0.5).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    # ------------------------------------------------------------------ #
    # Deterministic evaluation (mean angles)
    # ------------------------------------------------------------------ #
    pc_tensor = env.pc_tensor
    coverages = []
    last_info = None

    for ep in range(1, args.episodes + 1):
        with torch.no_grad():
            angles = policy.deterministic(pc_tensor)  # [K,2]
        angles_np = angles.cpu().numpy()

        cov, info = env.evaluate_viewset(angles_np, debug=(ep == 1))
        coverages.append(cov)
        last_info = info

        print(f"[Ep {ep:03d}] coverage={cov:.4f}")

    coverages = np.array(coverages, dtype=np.float32)
    print(f"\n[EVAL] mean coverage = {coverages.mean():.4f}, "
          f"max = {coverages.max():.4f}, min = {coverages.min():.4f}")

    # ------------------------------------------------------------------ #
    # Optional visualization of the last deterministic view-set
    # ------------------------------------------------------------------ #
    if args.visualize and last_info is not None:
        viewpoints = last_info["viewpoints"]
        env.visualize_viewset(viewpoints)


if __name__ == "__main__":
    main()
