#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import torch

from viewformer_model import ViewFormerPlanner
from visioncraft_data import load_single_model_points, evaluate_plan_on_model_single


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings-path", type=str, default="../../build/python_bindings")
    parser.add_argument("--model", type=str, default="../../models/gorilla.ply")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to viewformer_ep_xxxxxx.pt")
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--num-samples-pointcloud", type=int, default=250000)
    parser.add_argument("--resolution", type=float, default=-1.0)
    parser.add_argument("--num-views", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.bindings_path))
    from visioncraft_py import Model, Viewpoint, VisibilityManager  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)

    pts_norm, center, radius, model = load_single_model_points(
        model_path=args.model,
        num_points=args.num_points,
        num_samples_pointcloud=args.num_samples_pointcloud,
        resolution=args.resolution,
        Model=Model,
    )
    pts_batch = pts_norm.unsqueeze(0).to(device)

    # load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
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
        init_log_std=-0.5,
        in_point_dim=3,
        dropout=0.0,
    ).to(device)
    planner.load_state_dict(ckpt["model_state"])
    planner.eval()

    with torch.no_grad():
        raw_actions = planner.deterministic_actions(pts_batch)[0]

    reward, metrics = evaluate_plan_on_model_single(
        planner=planner,
        raw_actions=raw_actions,
        model=model,
        center=center,
        radius=radius,
        num_views=args.num_views,
        VisibilityManager=VisibilityManager,
        Viewpoint=Viewpoint,
        # use the same hyperparams you trained with:
        r_min_factor=6.5,
        r_max_factor=8.5,
        coverage_target=0.99,
        lambda_views=0.0,
        alpha_shortfall=0.0,
    )

    print("Deterministic evaluation:")
    print(f"  coverage = {metrics['coverage']:.4f}")
    print(f"  num_views_used = {metrics['num_views_used']}")
    print(f"  avg_radius = {metrics['avg_radius']:.2f}")
    print(f"  avg_gate   = {metrics['avg_gate']:.3f}")


if __name__ == "__main__":
    main()
