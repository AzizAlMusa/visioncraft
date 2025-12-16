# visualize_viewset_rl_single_model.py
#
# Load a trained viewset policy with gating, evaluate deterministically
# (mean directions + gated subset), and visualize viewpoints with Visioncraft's Visualizer.

import os
import sys
import argparse
import numpy as np
import torch

from viewset_policy import ViewSetPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings-path", type=str, default="../../build/python_bindings",
                        help="Path to directory containing visioncraft_py module")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to .ply model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint from training")
    parser.add_argument("--num-views", type=int, default=8,
                        help="Maximum number of candidate viewpoints (must match training)")
    parser.add_argument("--pc-size", type=int, default=2048,
                        help="PC size (should match training)")
    parser.add_argument("--radius", type=float, default=400.0,
                        help="Sphere radius (should match training)")
    parser.add_argument("--gate-threshold", type=float, default=0.5,
                        help="Gate probability threshold to keep a view at visualization time")
    parser.add_argument("--min-views", type=int, default=1,
                        help="Minimum number of views to enforce at visualization time")
    args = parser.parse_args()

    sys.path.append(args.bindings_path)
    from visioncraft_py import Model, Viewpoint, VisibilityManager, Visualizer  # noqa: E402
    from viewset_env import ViewSetEnv  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[VIS] Using device:", device)

    # --- Load checkpoint (for weights and, if available, config) ---
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", None)
    if cfg is not None:
        # Override CLI defaults with training config where available
        if "num_views" in cfg:
            args.num_views = cfg["num_views"]
        if "pc_size" in cfg:
            args.pc_size = cfg["pc_size"]
        if "radius" in cfg:
            args.radius = cfg["radius"]
        print("[VIS] Loaded training config from checkpoint: "
              "num_views={}, pc_size={}, radius={}".format(
                  args.num_views, args.pc_size, args.radius))
    else:
        print("[VIS] No config found in checkpoint; using CLI values.")

    # --- Env (for PC + coverage evaluation) ---
    env = ViewSetEnv(
        model_path=args.model,
        Model=Model,
        Viewpoint=Viewpoint,
        VisibilityManager=VisibilityManager,
        num_views=args.num_views,
        radius=args.radius,
        pc_size=args.pc_size,
        device=device,
        debug=True,
    )

    pc = env.get_state()

    # --- Load policy ---
    net = ViewSetPolicy(num_views=args.num_views).to(device)
    if "model_state_dict" not in ckpt:
        raise RuntimeError("Checkpoint missing 'model_state_dict': {}".format(args.checkpoint))
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    with torch.no_grad():
        mean_dirs, gate_logits = net(pc)  # (K_max,3), (K_max,)
        gate_probs = torch.sigmoid(gate_logits)

        # Deterministic gating: threshold, with a fallback to top-k if too few
        gate_mask = gate_probs > args.gate_threshold
        num_on = int(gate_mask.sum().item())
        if num_on < args.min_views:
            k = min(args.min_views, gate_probs.shape[0])
            _, top_idx = torch.topk(gate_probs, k=k)
            gate_mask = torch.zeros_like(gate_probs, dtype=torch.bool)
            gate_mask[top_idx] = True
            num_on = k

        selected_dirs = mean_dirs[gate_mask]
        print("[VIS] Using {} views out of max {}.".format(num_on, args.num_views))
        print("[VIS] Gate probabilities:", gate_probs.detach().cpu().numpy())

        cov_det, viewpoints = env.evaluate_viewset(
            selected_dirs, debug=True, return_viewpoints=True
        )

    print("[VIS] Deterministic coverage (gated) = {:.4f}".format(cov_det))

    # --- Visualize viewpoints and voxel visibility ---
    model = env.model  # same Model instance
    vis = Visualizer()
    vis.initializeWindow("Viewset RL Visualization (Gating)")
    vis.setBackgroundColor([0.0, 0.0, 0.0])

    # Optional: color voxels by visibility
    model.addVoxelProperty("visibility", 0)
    for vp in viewpoints:
        vis.addViewpoint(vp, True, True)
    vis.addVoxelMapProperty(model, "visibility", [1.0, 1.0, 1.0], [0.0, 1.0, 0.0])

    print("Press Ctrl+C to exit viewer.")
    try:
        import time
        while True:
            vis.renderStep()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Exited visualization.")


if __name__ == "__main__":
    main()
