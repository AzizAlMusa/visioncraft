# train_viewformer_cem_single_model.py

import os
import sys
import argparse
import numpy as np

import torch
import torch.optim as optim

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
    parser.add_argument("--episodes", type=int, default=300,
                        help="Number of CEM iterations")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Number of sampled plans per iteration")
    parser.add_argument("--num-views", type=int, default=6,
                        help="Number of views (K) used per plan")
    parser.add_argument("--pc-size", type=int, default=2048,
                        help="Number of points in normalized point cloud")
    parser.add_argument("--fixed-radius", type=float, default=400.0,
                        help="Radius of camera sphere in world units")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for Adam")
    parser.add_argument("--sigma-init", type=float, default=0.5,
                        help="Initial Gaussian noise std for actions")
    parser.add_argument("--sigma-min", type=float, default=0.05,
                        help="Minimum noise std")
    parser.add_argument("--sigma-decay", type=float, default=0.99,
                        help="Multiplicative decay per iteration")
    parser.add_argument("--out-dir", type=str,
                        default="outputs/viewformer_cem_single_model",
                        help="Output directory for checkpoints/metrics")
    parser.add_argument("--save-interval", type=int, default=100,
                        help="Save checkpoint every N iterations")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    #  Bindings
    # -------------------------------------------------------------------------
    sys.path.append(args.bindings_path)
    from visioncraft_py import Model, Viewpoint, VisibilityManager  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    #  Load model into Visioncraft
    # -------------------------------------------------------------------------
    model = Model()
    ok = model.loadModel(args.model, 250000)  # builds all voxel structures, normals, etc. :contentReference[oaicite:7]{index=7}
    if not ok:
        raise RuntimeError(f"Model.loadModel failed for {args.model}")

    # Normalized point cloud for network input (voxel space)
    pts_norm = build_normalized_pointcloud_from_voxels(model, num_points=args.pc_size)
    pts_t = torch.from_numpy(pts_norm).to(device)

    center_world, radius_world = get_world_center_and_radius(model)
    print(f"[INFO] Normalized PC shape: {pts_t.shape}, radius(world)={radius_world:.4f}")

    # -------------------------------------------------------------------------
    #  Build planner
    # -------------------------------------------------------------------------
    planner = ViewFormerPlanner(num_views=args.num_views, per_view_dim=2)
    planner.to(device)

    optimizer = optim.Adam(planner.parameters(), lr=args.lr)

    sigma = args.sigma_init

    best_cov_history = []
    mean_cov_history = []
    det_cov_history = []
    sigma_history = []

    # -------------------------------------------------------------------------
    #  CEM loop
    # -------------------------------------------------------------------------
    for ep in range(1, args.episodes + 1):
        # 1) Get current deterministic action mean mu(P)
        with torch.no_grad():
            raw_mu = planner(pts_t)           # (K,2)
            raw_mu_np = raw_mu.cpu().numpy()  # (K,2)

        # 2) Sample batch of noisy actions and evaluate
        batch_cov = []
        batch_actions = []

        for b in range(args.batch_size):
            noise = np.random.randn(*raw_mu_np.shape).astype(np.float32)
            raw_actions = raw_mu_np + sigma * noise  # (K,2)

            cov = evaluate_sphere_plan_single_model(
                raw_actions=raw_actions,
                model=model,
                center_world=center_world,
                radius_world=radius_world,
                num_views=args.num_views,
                VisibilityManager=VisibilityManager,
                Viewpoint=Viewpoint,
                fixed_radius=args.fixed_radius,
            )

            batch_cov.append(cov)
            batch_actions.append(raw_actions)

        batch_cov = np.asarray(batch_cov, dtype=np.float32)       # (M,)
        batch_actions = np.stack(batch_actions, axis=0)           # (M,K,2)

        best_idx = int(batch_cov.argmax())
        best_cov = float(batch_cov[best_idx])
        mean_cov = float(batch_cov.mean())

        # 3) Elite selection
        elite_frac = 0.2
        elite_count = max(1, int(elite_frac * args.batch_size))
        elite_idx = np.argsort(batch_cov)[-elite_count:]

        elite_actions = batch_actions[elite_idx]       # (E,K,2)
        elite_cov = batch_cov[elite_idx]              # (E,)

        elite_mean_actions = elite_actions.mean(axis=0)  # (K,2)

        # 4) Deterministic coverage of current mean (no noise)
        det_cov = evaluate_sphere_plan_single_model(
            raw_actions=raw_mu_np,
            model=model,
            center_world=center_world,
            radius_world=radius_world,
            num_views=args.num_views,
            VisibilityManager=VisibilityManager,
            Viewpoint=Viewpoint,
            fixed_radius=args.fixed_radius,
        )

        # 5) Fit mu(P) to elite_mean_actions
        elite_mean_t = torch.from_numpy(elite_mean_actions).float().to(device)

        optimizer.zero_grad()
        raw_mu_new = planner(pts_t)  # (K,2)
        loss = torch.mean((raw_mu_new - elite_mean_t) ** 2)
        loss.backward()
        optimizer.step()

        # 6) Update sigma (simple annealing)
        sigma = max(args.sigma_min, sigma * args.sigma_decay)

        # Logging
        best_cov_history.append(best_cov)
        mean_cov_history.append(mean_cov)
        det_cov_history.append(det_cov)
        sigma_history.append(sigma)

        if ep % 10 == 1 or ep == args.episodes:
            print(
                f"[Ep {ep:04d}] "
                f"best={best_cov:.4f} mean={mean_cov:.4f} det={det_cov:.4f} "
                f"sigma={sigma:.3f} loss={loss.item():.4f}"
            )

        # Save checkpoint
        if ep % args.save_interval == 0 or ep == args.episodes:
            ckpt_path = os.path.join(args.out_dir, f"viewformer_cem_ep_{ep:06d}.pt")
            torch.save(
                {
                    "planner_state": planner.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "sigma": sigma,
                    "episode": ep,
                },
                ckpt_path,
            )
            print(f"[SAVE] Checkpoint -> {ckpt_path}")

            metrics_path = os.path.join(args.out_dir, "training_metrics_cem.npz")
            np.savez_compressed(
                metrics_path,
                best_cov=np.asarray(best_cov_history, dtype=np.float32),
                mean_cov=np.asarray(mean_cov_history, dtype=np.float32),
                det_cov=np.asarray(det_cov_history, dtype=np.float32),
                sigma=np.asarray(sigma_history, dtype=np.float32),
            )
            print(f"[SAVE] Metrics -> {metrics_path}")

    print("Training finished.")
    if len(best_cov_history) > 0:
        print(f"Final: best={best_cov_history[-1]:.4f}, det={det_cov_history[-1]:.4f}")


if __name__ == "__main__":
    main()
