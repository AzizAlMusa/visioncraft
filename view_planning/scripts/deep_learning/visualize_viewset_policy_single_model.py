#!/usr/bin/env python3
import argparse
import os
import sys
import time

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings-path", type=str, required=True,
                        help="Path to visioncraft_py python_bindings directory")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to .ply model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained viewset RL checkpoint (.pt)")
    parser.add_argument("--num-views", type=int, default=4,
                        help="Number of views the network was trained with")
    parser.add_argument("--pc-size", type=int, default=2048)
    parser.add_argument("--r-min", type=float, default=300.0)
    parser.add_argument("--r-max", type=float, default=500.0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Paths + imports
    # ------------------------------------------------------------------
    sys.path.append(args.bindings_path)
    from visioncraft_py import Model, Viewpoint, VisibilityManager, Visualizer  # type: ignore

    from viewset_env import ViewSetEnv
    from viewset_policy import ViewSetPolicyNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[VIS] Using device:", device)

    # ------------------------------------------------------------------
    # Load model + env
    # ------------------------------------------------------------------
    model = Model()
    ok = model.loadModel(args.model, 250000)
    if not ok:
        raise RuntimeError("Failed to load model: %s" % args.model)

    env = ViewSetEnv(model,
                     ViewpointCls=Viewpoint,
                     VisibilityManagerCls=VisibilityManager,
                     num_views=args.num_views,
                     r_min=args.r_min,
                     r_max=args.r_max,
                     pc_size=args.pc_size,
                     debug=args.debug)

    pc_norm = torch.from_numpy(env.pc_norm).to(device)

    # ------------------------------------------------------------------
    # Load policy
    # ------------------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location=device)
    num_views_ckpt = ckpt.get("num_views", args.num_views)

    if num_views_ckpt != args.num_views:
        print("[WARN] Checkpoint was trained with num_views=%d, "
              "but you passed --num-views=%d"
              % (num_views_ckpt, args.num_views))

    net = ViewSetPolicyNet(num_views=num_views_ckpt).to(device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    # ------------------------------------------------------------------
    # Deterministic evaluation: use mean action (no sampling)
    # ------------------------------------------------------------------
    with torch.no_grad():
        mean_action, _ = net(pc_norm)

    action_np = mean_action.cpu().numpy()
    dirs, radii = env.actions_to_dirs_and_radii(action_np)

    # Evaluate once with debug + get viewpoints and manager
    coverage, viewpoints, vis_manager = env.evaluate_viewset(
        dirs, radii,
        debug=True,
        return_viewpoints=True,
        return_manager=True
    )

    print("[VIS] Deterministic coverage = %.4f" % coverage)

    # ------------------------------------------------------------------
    # Visualize using Visioncraft Visualizer
    # ------------------------------------------------------------------
    vis = Visualizer()
    vis.initializeWindow("Viewset RL - Visualization")
    vis.setBackgroundColor([0.0, 0.0, 0.0])

    # Color voxels by "visibility" property set by vis_manager
    vis.addVoxelMapProperty(model, "visibility",
                            [1.0, 1.0, 1.0],   # base color
                            [0.0, 1.0, 0.0])   # visible voxels

    for vp in viewpoints:
        vis.addViewpoint(vp, True, True)

    print("Press Ctrl+C to exit viewer.")
    try:
        while True:
            vis.renderStep()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Exited.")


if __name__ == "__main__":
    main()
