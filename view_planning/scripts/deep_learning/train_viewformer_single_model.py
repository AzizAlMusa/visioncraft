#!/usr/bin/env python3
import argparse
import os
import sys
import random
import traceback

import numpy as np
import torch
import torch.optim as optim

from viewformer_model import ViewFormerPlanner
from visioncraft_data import (
    load_single_model_points,
    evaluate_plan_on_model_single,
    TrainingLogger,
    EpisodeMetrics,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_single_model(args):
    # --------------------------------
    # Import Visioncraft bindings
    # --------------------------------
    sys.path.insert(0, os.path.abspath(args.bindings_path))
    try:
        from visioncraft_py import Model, Viewpoint, VisibilityManager  # type: ignore
        print("[OK] Imported Model, Viewpoint, VisibilityManager from visioncraft_py")
    except Exception as e:
        print("[FAIL] Could not import visioncraft_py:", e)
        traceback.print_exc()
        return

    if not os.path.isfile(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)

    set_seed(args.seed)

    # --------------------------------
    # Load model + point cloud (once)
    # --------------------------------
    pts_norm, center, radius, model = load_single_model_points(
        model_path=args.model,
        num_points=args.num_points,
        num_samples_pointcloud=args.num_samples_pointcloud,
        resolution=args.resolution,
        Model=Model,
    )
    print(f"[INFO] Normalized point cloud shape: {pts_norm.shape}, radius(world)={radius:.4f}")

    pts_batch = pts_norm.unsqueeze(0).to(device)  # (1,N,3)

    # --------------------------------
    # Initialize planner + optimizer
    # --------------------------------
    planner = ViewFormerPlanner(
        num_views=args.num_views,
        per_view_dim=8,
        point_feat_dim=256,
        view_token_dim=256,
        view_emb_dim=256,
        num_point_layers=2,
        num_point_heads=4,
        num_view_layers=1,
        num_view_heads=4,
        init_log_std=args.init_log_std,
        in_point_dim=3,
        dropout=args.dropout,
    ).to(device)

    optimizer = optim.Adam(planner.parameters(), lr=args.lr)
    value_loss_coef = args.value_loss_coef

    # --------------------------------
    # Logger for analysis
    # --------------------------------
    logger = TrainingLogger()

    # --------------------------------
    # Simple baseline for advantage
    # --------------------------------
    baseline = 0.0
    baseline_momentum = 0.9

    print("\n=== Training ViewFormer on single model ===")
    for ep in range(1, args.episodes + 1):
        planner.train()

        # Sample action from policy
        raw_actions, log_probs, values = planner.sample_actions(pts_batch)
        raw_actions_0 = raw_actions[0]
        log_prob_0 = log_probs[0]
        value_0 = values[0]

        # Evaluate plan via Visioncraft
        fixed_radius_world = args.fixed_radius if args.fixed_radius > 0 else None

        reward, metrics = evaluate_plan_on_model_single(
            planner=planner,
            raw_actions=raw_actions_0,
            model=model,
            center=center,
            radius=radius,
            num_views=args.num_views,
            VisibilityManager=VisibilityManager,
            Viewpoint=Viewpoint,
            r_min_factor=args.r_min_factor,
            r_max_factor=args.r_max_factor,
            coverage_target=args.coverage_target,
            lambda_views=args.lambda_views,
            alpha_shortfall=args.alpha_shortfall,
            downsample=args.downsample_factor,
            near=args.near_plane,
            far=args.far_plane,
            hfov=args.hfov,
            vfov=args.vfov,
            fixed_radius_world=fixed_radius_world,
            use_spherical_lookat=args.sphere_lookat,
            use_gating=args.use_gating,
        )



        reward_t = torch.tensor(reward, dtype=torch.float32, device=device)
        advantage = reward_t - value_0

        policy_loss = -(log_prob_0 * advantage.detach())
        value_loss = advantage.pow(2)
        loss = policy_loss + value_loss_coef * value_loss

        optimizer.zero_grad()
        loss.backward()
        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(planner.parameters(), args.max_grad_norm)
        optimizer.step()

        # update moving baseline for logging (not used in loss here but can be)
        baseline = baseline_momentum * baseline + (1.0 - baseline_momentum) * reward

        # log metrics
        ep_metrics = EpisodeMetrics(
            episode=ep,
            reward=float(reward),
            coverage=float(metrics["coverage"]),
            num_views_used=int(metrics["num_views_used"]),
            avg_gate=float(metrics["avg_gate"]),
            avg_radius=float(metrics["avg_radius"]),
        )
        logger.log(ep_metrics)

        if ep % args.log_interval == 0 or ep == 1:
            print(
                f"[Ep {ep:04d}] "
                f"reward={ep_metrics.reward:.4f} "
                f"coverage={ep_metrics.coverage:.4f} "
                f"views_used={ep_metrics.num_views_used} "
                f"avg_gate={ep_metrics.avg_gate:.3f} "
                f"loss={loss.item():.4f} "
                f"baseline={baseline:.4f}"
            )
            print("    ", logger.summary_str(last_n=min(ep, args.log_interval)))

        if ep % args.save_interval == 0 or ep == args.episodes:
            # save weights
            os.makedirs(args.out_dir, exist_ok=True)
            ckpt_path = os.path.join(args.out_dir, f"viewformer_ep_{ep:06d}.pt")
            torch.save(
                {
                    "episode": ep,
                    "model_state": planner.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"[SAVE] Checkpoint -> {ckpt_path}")

            # save metrics for analysis
            metrics_path = os.path.join(args.out_dir, "training_metrics.npz")
            logger.save_npz(metrics_path)
            print(f"[SAVE] Metrics -> {metrics_path}")

    print("\nTraining finished.")
    print("Final summary:", logger.summary_str(last_n=min(args.episodes, 50)))

    # If you want programmatic access (e.g., from Jupyter), you can
    # return logger and planner here instead of running as a script.
    return planner, logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bindings-path",
        type=str,
        default="../../build/python_bindings",
        help="Path to directory containing visioncraft_py bindings",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="../../models/gorilla.ply",
        help="Path to a single test model (.ply)",
    )

    # geometry / point cloud sampling
    parser.add_argument("--num-points", type=int, default=2048,
                        help="Number of voxel centers sampled for network input")
    parser.add_argument("--num-samples-pointcloud", type=int, default=250000,
                        help="num_samples for Model.loadModel")
    parser.add_argument("--resolution", type=float, default=-1.0,
                        help="resolution for Model.loadModel (<=0 means auto)")

    # planner config
    parser.add_argument("--num-views", type=int, default=8,
                        help="Number of view slots K in the plan")
    parser.add_argument("--init-log-std", type=float, default=-0.5)
    parser.add_argument("--dropout", type=float, default=0.0)

    # RL hyperparams
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # reward shaping
    parser.add_argument("--coverage-target", type=float, default=0.99)
    parser.add_argument("--lambda-views", type=float, default=0.05)
    parser.add_argument("--alpha-shortfall", type=float, default=10.0)
    parser.add_argument("--r-min-factor", type=float, default=1.5)
    parser.add_argument("--r-max-factor", type=float, default=2.5)

    parser.add_argument("--fixed-radius", type=float, default=400.0,
                        help="If >0 and --sphere-lookat is set, constrain cameras to this world-space radius")

    # camera mode
    parser.add_argument("--sphere-lookat", action="store_true",
                        help="Constrain cameras on fixed-radius sphere and look at object center")
    parser.add_argument("--use-gating", action="store_true",
                        help="Use gate outputs to decide which views are active (default: off)")
    # camera frustum
    parser.add_argument("--near-plane", type=float, default=300.0)
    parser.add_argument("--far-plane", type=float, default=900.0)
    parser.add_argument("--hfov", type=float, default=44.8)
    parser.add_argument("--vfov", type=float, default=42.6)
    parser.add_argument("--downsample-factor", type=float, default=4.0)

    # logging / output
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--out-dir", type=str, default="outputs/viewformer_single_model")

    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    train_single_model(args)


if __name__ == "__main__":
    main()
