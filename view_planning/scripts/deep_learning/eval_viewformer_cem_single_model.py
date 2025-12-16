# eval_viewformer_cem_single_model.py

import os
import sys
import argparse
import numpy as np
import torch

from viewformer_cem_model import ViewFormerPlanner
from cem_env_single_model import (
    build_normalized_pointcloud_from_voxels,
    get_world_center_and_radius,
    evaluate_sphere_plan_single_model,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings-path", type=str, required=True,
                        help="Path to directory containing visioncraft_py bindings")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to CAD model (.ply)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint .pt from CEM training")
    parser.add_argument("--num-views", type=int, default=6,
                        help="Number of views (K) used by the planner")
    parser.add_argument("--pc-size", type=int, default=2048,
                        help="Number of points in normalized point cloud")
    parser.add_argument("--fixed-radius", type=float, default=400.0,
                        help="Camera sphere radius in world units")
    args = parser.parse_args()

    sys.path.append(args.bindings_path)
    from visioncraft_py import Model, Viewpoint, VisibilityManager  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = Model()
    ok = model.loadModel(args.model, 250000)
    if not ok:
        raise RuntimeError(f"Model.loadModel failed for {args.model}")

    pts_norm = build_normalized_pointcloud_from_voxels(model, num_points=args.pc_size)
    pts_t = torch.from_numpy(pts_norm).to(device)

    center_world, radius_world = get_world_center_and_radius(model)

    # Build planner and load checkpoint
    planner = ViewFormerPlanner(num_views=args.num_views, per_view_dim=2)
    planner.to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    planner.load_state_dict(ckpt["planner_state"])
    sigma = ckpt.get("sigma", None)
    ep = ckpt.get("episode", None)
    print(f"[LOAD] {args.checkpoint} (episode={ep}, sigma={sigma})")

    planner.eval()
    with torch.no_grad():
        raw_mu = planner(pts_t)           # (K,2)
        raw_mu_np = raw_mu.cpu().numpy()  # (K,2)

    coverage = evaluate_sphere_plan_single_model(
        raw_actions=raw_mu_np,
        model=model,
        center_world=center_world,
        radius_world=radius_world,
        num_views=args.num_views,
        VisibilityManager=VisibilityManager,
        Viewpoint=Viewpoint,
        fixed_radius=args.fixed_radius,
    )

    print("Deterministic evaluation:")
    print(f"  coverage      = {coverage:.4f}")
    print(f"  num_views     = {args.num_views}")
    print(f"  fixed_radius  = {args.fixed_radius:.2f}")


if __name__ == "__main__":
    main()
