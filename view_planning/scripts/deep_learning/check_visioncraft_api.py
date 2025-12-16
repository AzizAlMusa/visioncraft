#!/usr/bin/env python3
import argparse
import os
import sys
import math
import random
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# -------------------------------
# PyTorch model
# -------------------------------

class PointNetEncoder(nn.Module):
    """Minimal PointNet-style encoder: (B, N, 3) -> (B, F)"""

    def __init__(self, input_dim: int = 3, feat_dim: int = 256):
        super().__init__()
        self.mlp1 = nn.Linear(input_dim, 64)
        self.mlp2 = nn.Linear(64, 128)
        self.mlp3 = nn.Linear(128, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 3)
        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        x = self.mlp3(x)
        # symmetric pooling
        x = torch.max(x, dim=1).values  # (B, F)
        return x


class SingleShotPlanner(nn.Module):
    """
    Point-cloud -> multi-view plan.

    - Input: (B, N, 3) normalized point clouds.
    - Output: Gaussian policy over raw action vector a ∈ R^{B x (K * per_view_dim)}.
      per_view_dim = 8: [dir3, radius1, yaw/pitch/roll3, gate1].
    """

    def __init__(
        self,
        num_views: int = 16,
        per_view_dim: int = 8,
        point_feat_dim: int = 256,
        view_emb_dim: int = 64,
        init_log_std: float = 0.0,
    ):
        super().__init__()
        self.num_views = num_views
        self.per_view_dim = per_view_dim
        self.action_dim = num_views * per_view_dim

        self.encoder = PointNetEncoder(input_dim=3, feat_dim=point_feat_dim)

        # K learnable view tokens
        self.view_embeddings = nn.Parameter(
            torch.randn(num_views, view_emb_dim) * 0.01
        )

        # Actor head
        self.actor_mlp1 = nn.Linear(point_feat_dim + view_emb_dim, 256)
        self.actor_mlp2 = nn.Linear(256, 256)
        self.actor_out = nn.Linear(256, per_view_dim)

        # Critic head
        self.critic_mlp1 = nn.Linear(point_feat_dim, 256)
        self.critic_mlp2 = nn.Linear(256, 256)
        self.critic_out = nn.Linear(256, 1)

        # Global log-std for all action dims
        self.log_std = nn.Parameter(
            torch.ones(1, self.action_dim) * init_log_std
        )

    def forward_encoder(self, points: torch.Tensor) -> torch.Tensor:
        return self.encoder(points)

    def forward_actor(self, global_feat: torch.Tensor) -> torch.Tensor:
        B = global_feat.shape[0]
        # expand global_feat over K view slots
        g_exp = global_feat.unsqueeze(1).expand(-1, self.num_views, -1)  # (B,K,F)
        v_emb = self.view_embeddings.unsqueeze(0).expand(B, -1, -1)      # (B,K,E)
        x = torch.cat([g_exp, v_emb], dim=-1)                             # (B,K,F+E)
        x = F.relu(self.actor_mlp1(x))
        x = F.relu(self.actor_mlp2(x))
        per_view_mu = self.actor_out(x)                                   # (B,K,Dv)
        mu = per_view_mu.reshape(B, -1)                                   # (B, A)
        return mu

    def forward_critic(self, global_feat: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.critic_mlp1(global_feat))
        x = F.relu(self.critic_mlp2(x))
        v = self.critic_out(x).squeeze(-1)  # (B,)
        return v

    def forward(self, points: torch.Tensor):
        g = self.forward_encoder(points)
        mu = self.forward_actor(g)
        v = self.forward_critic(g)
        return mu, v

    def sample_actions(self, points: torch.Tensor):
        """
        Sample raw actions from Gaussian policy.

        Returns:
            raw_actions: (B, A)
            log_probs:   (B,)
            values:      (B,)
        """
        mu, values = self.forward(points)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        raw_actions = dist.rsample()
        log_probs = dist.log_prob(raw_actions).sum(dim=-1)
        return raw_actions, log_probs, values

    def decode_raw_actions(
        self,
        raw_actions: torch.Tensor,
        r_min: float,
        r_max: float,
        center: torch.Tensor,
    ):
        """
        Decode raw actions into SE(3) poses.

        raw_actions: (B, A)
        center:      (B, 3) world-space centers
        returns:
            positions: (B, K, 3)
            ypr:       (B, K, 3) yaw,pitch,roll in radians
            gates:     (B, K)
        """
        B = raw_actions.shape[0]
        per = self.per_view_dim
        K = self.num_views

        x = raw_actions.view(B, K, per)

        dir_raw = x[..., 0:3]   # (B,K,3)
        rad_raw = x[..., 3:4]   # (B,K,1)
        ypr_raw = x[..., 4:7]   # (B,K,3)
        gate_raw = x[..., 7]    # (B,K,)

        # directions
        dir_norm = torch.norm(dir_raw, dim=-1, keepdim=True).clamp(min=1e-6)
        dirs = dir_raw / dir_norm

        # radii in [r_min,r_max]
        radii_norm = torch.sigmoid(rad_raw)
        radii = r_min + radii_norm * (r_max - r_min)

        # positions
        center_exp = center.unsqueeze(1)     # (B,1,3)
        positions = center_exp + radii * dirs

        # yaw/pitch/roll in (-pi,pi)
        ypr = math.pi * torch.tanh(ypr_raw)

        # gates in (0,1)
        gates = torch.sigmoid(gate_raw)

        return positions, ypr, gates


# -------------------------------
# Visioncraft interop helpers
# -------------------------------

def load_pointcloud_from_model(model_path: str,
                               num_points: int,
                               num_samples_pointcloud: int,
                               resolution: float,
                               Model):
    """
    Load Model, extract surface voxel centers as point cloud, normalize.

    Returns:
        pts_norm: (N,3) torch float32
        center:   np.ndarray (3,) world coords
        radius:   float (max distance from center)
        model:    Model instance (with voxel structures ready)
    """
    model = Model()
    ok = model.loadModel(model_path, num_samples_pointcloud, resolution)
    if not ok:
        print("[WARN] loadModel returned False for", model_path)

    # get voxel centers via MetaVoxel
    voxel_keys = list(model.getVoxelMap())
    if not voxel_keys:
        raise RuntimeError("No voxels returned by getVoxelMap()")

    positions = []
    if hasattr(model, "getVoxel"):
        for key in voxel_keys:
            mv = model.getVoxel(key)
            if mv is None:
                continue
            pos = mv.getPosition()  # Eigen::Vector3d
            positions.append([pos[0], pos[1], pos[2]])
    else:
        raise RuntimeError("Model has no getVoxel(key) binding; cannot get voxel centers")

    positions = np.asarray(positions, dtype=np.float32)  # (M,3)

    # center & radius
    center = positions.mean(axis=0)
    centered = positions - center
    radii = np.linalg.norm(centered, axis=1)
    radius = float(radii.max())

    # normalize to roughly unit sphere
    pts_norm = centered / (radius + 1e-6)

    # subsample to fixed N
    M = pts_norm.shape[0]
    if M >= num_points:
        idx = np.random.choice(M, num_points, replace=False)
    else:
        pad = np.random.choice(M, num_points - M, replace=True)
        idx = np.concatenate([np.arange(M), pad], axis=0)

    pts_norm = pts_norm[idx]
    pts_norm_torch = torch.from_numpy(pts_norm).float()  # (N,3)

    return pts_norm_torch, center, radius, model


def evaluate_plan_on_model(
    model,
    planner: SingleShotPlanner,
    raw_actions: torch.Tensor,
    center: np.ndarray,
    radius: float,
    num_views: int,
    r_min_factor: float = 1.5,
    r_max_factor: float = 2.5,
    coverage_target: float = 0.99,
    lambda_views: float = 0.05,
    alpha_shortfall: float = 10.0,
    downsample: float = 4.0,
    near: float = 300.0,
    far: float = 900.0,
    hfov: float = 44.8,
    vfov: float = 42.6,
    VisibilityManager=None,
    Viewpoint=None,
):
    """
    Decode raw_actions into viewpoints, run GPU raycasting,
    compute coverage and a scalar reward.

    raw_actions: (A,) tensor on same device as planner.
    center:      (3,) numpy center in world coords.
    """
    device = next(planner.parameters()).device

    raw_actions = raw_actions.unsqueeze(0).to(device)  # (1,A)
    center_t = torch.from_numpy(center.astype(np.float32)).unsqueeze(0).to(device)

    r_min = r_min_factor * radius
    r_max = r_max_factor * radius

    positions, ypr, gates = planner.decode_raw_actions(
        raw_actions, r_min=r_min, r_max=r_max, center=center_t
    )

    positions = positions[0].detach().cpu().numpy()  # (K,3)
    ypr = ypr[0].detach().cpu().numpy()              # (K,3)
    gates = gates[0].detach().cpu().numpy()          # (K,)

    vm = VisibilityManager(model)

    used = 0
    gate_thresh = 0.5

    for i in range(num_views):
        if gates[i] < gate_thresh:
            continue
        used += 1
        pos = positions[i]
        yaw, pitch, roll = ypr[i]
        vec = np.array([pos[0], pos[1], pos[2], yaw, pitch, roll], dtype=np.float64)

        vp = Viewpoint.from_euler(
            vec, near, far, 2448, 2048, hfov, vfov
        )
        vp.setDownsampleFactor(downsample)

        vm.trackViewpoint(vp)
        vp.performRaycastingOnGPU(model)

    coverage = float(vm.getCoverageScore())

    shortfall = max(0.0, coverage_target - coverage)
    reward = coverage - lambda_views * (used / max(1, num_views)) - alpha_shortfall * (shortfall ** 2)

    return reward, coverage, used


# -------------------------------
# Mock RL loop
# -------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings-path", type=str, default="../../build/python_bindings",
                        help="Path to directory containing visioncraft_py bindings")
    parser.add_argument("--model", type=str, default="../../models/gorilla.ply",
                        help="Path to a test .ply model")
    parser.add_argument("--num-points", type=int, default=2048,
                        help="Number of points sampled from voxel map for network input")
    parser.add_argument("--num-samples-pointcloud", type=int, default=250000,
                        help="num_samples parameter for Model.loadModel")
    parser.add_argument("--resolution", type=float, default=-1.0,
                        help="resolution parameter for Model.loadModel")
    parser.add_argument("--num-views", type=int, default=8,
                        help="Number of view slots K in the planner")
    parser.add_argument("--episodes", type=int, default=10,
                        help="How many mock RL episodes to run")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    # import Visioncraft bindings
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

    planner = SingleShotPlanner(
        num_views=args.num_views,
        per_view_dim=8,
        point_feat_dim=256,
        view_emb_dim=64,
        init_log_std=0.0,
    ).to(device)

    optimizer = torch.optim.Adam(planner.parameters(), lr=args.lr)
    value_coef = 0.5

    print("\n=== Mock RL loop ===")
    for ep in range(1, args.episodes + 1):
        # Load a fresh model and its point cloud
        pts_norm, center, radius, model = load_pointcloud_from_model(
            args.model,
            num_points=args.num_points,
            num_samples_pointcloud=args.num_samples_pointcloud,
            resolution=args.resolution,
            Model=Model,
        )

        pts_batch = pts_norm.unsqueeze(0).to(device)  # (1,N,3)

        # Sample action and value from current policy
        raw_actions, log_probs, values = planner.sample_actions(pts_batch)
        raw_actions_0 = raw_actions[0]       # (A,)
        log_prob_0 = log_probs[0]           # scalar
        value_0 = values[0]                 # scalar

        # Evaluate the plan with Visioncraft
        reward, coverage, used = evaluate_plan_on_model(
            model=model,
            planner=planner,
            raw_actions=raw_actions_0,
            center=center,
            radius=radius,
            num_views=args.num_views,
            VisibilityManager=VisibilityManager,
            Viewpoint=Viewpoint,
        )

        reward_t = torch.tensor(reward, dtype=torch.float32, device=device)
        advantage = reward_t - value_0

        policy_loss = -(log_prob_0 * advantage.detach())
        value_loss = advantage.pow(2)
        loss = policy_loss + value_coef * value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"[Ep {ep:03d}] "
              f"reward={reward:.4f} "
              f"coverage={coverage:.4f} "
              f"used_views={used} "
              f"loss={loss.item():.4f}")

    print("\nDone. This proves:")
    print("  - we can turn voxel centers into a point cloud input for the net,")
    print("  - we can decode network outputs into Viewpoints,")
    print("  - Visioncraft’s GPU raycasting + VisibilityManager give us coverage,")
    print("  - and we can run a simple RL-style update loop end-to-end.")


if __name__ == "__main__":
    main()
