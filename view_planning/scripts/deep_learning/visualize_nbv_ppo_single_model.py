#!/usr/bin/env python3
import argparse
import sys
import time
import numpy as np
import torch
import torch.nn as nn

from nbv_env_discrete import build_env_from_visioncraft


# ---------------- Actor–Critic (must match training) ---------------- #

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, num_actions)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.policy_head(h), self.value_head(h)


# ------------------------ Main script ------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings-path", type=str, required=True,
                        help="Path to ../../build/python_bindings")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model .ply")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to PPO checkpoint .pt")
    parser.add_argument("--num-candidates", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[VIS] Using device: {device}")

    # --------- Build env (vis_matrix + candidate positions) --------- #
    env = build_env_from_visioncraft(
        bindings_path=args.bindings_path,
        model_path=args.model,
        num_candidates=args.num_candidates,
        max_steps=args.max_steps,
        fixed_radius=400.0,          # same as training
        downsample_factor=4.0,
        seed=0,
        verbose=True,
    )

    obs_dim = env.obs_dim
    num_actions = env.num_views

    # --------- Load policy --------- #
    net = ActorCritic(obs_dim, num_actions).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    # --------- Deterministic eval with unique actions --------- #
    def run_det_episode():
        obs = env.reset()
        used_actions = []
        coverage = 0.0

        for t in range(args.max_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = net(obs_t)

            # mask out already used views
            used_mask = obs[:num_actions] > 0.5   # obs layout: [used_mask, coverage, step_frac]
            logits = logits.clone()
            logits[0, used_mask] = -1e9

            action = int(torch.argmax(logits, dim=-1).item())
            used_actions.append(action)

            obs, reward, done, info = env.step(action)
            coverage = info["coverage"]
            if done:
                break

        return coverage, used_actions

    cov, actions = run_det_episode()
    print(f"[VIS] deterministic coverage = {cov:.4f}, actions = {actions}")

    # --------- 3D visualization with Visioncraft --------- #
    sys.path.append(args.bindings_path)
    from visioncraft_py import Model, Viewpoint, VisibilityManager, Visualizer

    model = Model()
    ok = model.loadModel(args.model, 250000)
    if not ok:
        raise RuntimeError(f"Model.loadModel failed for {args.model}")

    vm = VisibilityManager(model)

    visualizer = Visualizer()
    visualizer.initializeWindow("NBV PPO Viewpoints")
    visualizer.setBackgroundColor([0.0, 0.0, 0.0])

    center = np.asarray(model.getCenter(), dtype=np.float32)

    selected_viewpoints = []

    for a in actions:
        pos = env.candidate_positions[a]  # already in world coords
        vp = Viewpoint.from_lookat(pos.tolist(), center.tolist())
        vp.setNearPlane(300.0)
        vp.setFarPlane(900.0)
        vp.setDownsampleFactor(4.0)

        vm.trackViewpoint(vp)
        vp.performRaycastingOnGPU(model)
        selected_viewpoints.append(vp)

    # Add voxel visibility property (green = visible)
    visualizer.addVoxelMapProperty(model, "visibility",
                                   [1.0, 1.0, 1.0],   # base color
                                   [0.0, 1.0, 0.0])   # visible color

    for vp in selected_viewpoints:
        visualizer.addViewpoint(vp, True, True)

    print(f"[VIS] Coverage from VisibilityManager = {vm.getCoverageScore():.4f}")
    print("[VIS] Close the window (Ctrl+C) to exit.")

    try:
        while True:
            visualizer.renderStep()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n[VIS] Exiting viewer.")


if __name__ == "__main__":
    main()
