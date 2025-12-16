#!/usr/bin/env python3
import os
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, List

import numpy as np
import torch


@dataclass
class EpisodeMetrics:
    episode: int
    reward: float
    coverage: float
    num_views_used: int
    avg_gate: float
    avg_radius: float


class TrainingLogger:
    """
    Simple logger to store per-episode metrics and export them later
    for analysis/plotting.
    """

    def __init__(self):
        self.episodes: List[int] = []
        self.rewards: List[float] = []
        self.coverages: List[float] = []
        self.num_views_used: List[int] = []
        self.avg_gates: List[float] = []
        self.avg_radii: List[float] = []

    def log(self, metrics: EpisodeMetrics):
        self.episodes.append(metrics.episode)
        self.rewards.append(metrics.reward)
        self.coverages.append(metrics.coverage)
        self.num_views_used.append(metrics.num_views_used)
        self.avg_gates.append(metrics.avg_gate)
        self.avg_radii.append(metrics.avg_radius)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode": np.array(self.episodes, dtype=np.int32),
            "reward": np.array(self.rewards, dtype=np.float32),
            "coverage": np.array(self.coverages, dtype=np.float32),
            "num_views_used": np.array(self.num_views_used, dtype=np.int32),
            "avg_gate": np.array(self.avg_gates, dtype=np.float32),
            "avg_radius": np.array(self.avg_radii, dtype=np.float32),
        }

    def save_npz(self, path: str):
        data = self.to_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **data)

    def summary_str(self, last_n: int = 10) -> str:
        if len(self.rewards) == 0:
            return "no data"
        n = min(last_n, len(self.rewards))
        return (
            f"last {n} episodes: "
            f"reward={np.mean(self.rewards[-n:]):.4f}, "
            f"coverage={np.mean(self.coverages[-n:]):.4f}, "
            f"views={np.mean(self.num_views_used[-n:]):.2f}"
        )


def load_single_model_points(
    model_path: str,
    num_points: int,
    num_samples_pointcloud: int,
    resolution: float,
    Model,
):
    """
    Load a single CAD model through Visioncraft Model and extract
    surface-voxel centers as a point cloud.

    Returns:
        pts_norm: (N,3) torch float32, normalized (centered & scaled)
        center:   np.ndarray (3,) world-space center
        radius:   float, max distance to center (world units)
        model:    Model instance with voxel structures ready
    """
    model = Model()
    ok = model.loadModel(model_path, num_samples_pointcloud, resolution)
    if not ok:
        print(f"[WARN] loadModel returned False for {model_path}")

    # get voxel centers via MetaVoxel
    voxel_keys = list(model.getVoxelMap())
    if not voxel_keys:
        raise RuntimeError("No voxels from model.getVoxelMap()")

    positions = []
    if hasattr(model, "getVoxel"):
        for key in voxel_keys:
            mv = model.getVoxel(key)
            if mv is None:
                continue
            pos = mv.getPosition()  # Eigen::Vector3d
            positions.append([pos[0], pos[1], pos[2]])
    else:
        raise RuntimeError("Model has no getVoxel(key) binding; cannot fetch voxel centers")

    positions = np.asarray(positions, dtype=np.float32)  # (M,3)

    # compute center & radius (world coords)
    center = positions.mean(axis=0)
    centered = positions - center
    dists = np.linalg.norm(centered, axis=1)
    radius = float(dists.max())

    # normalized points
    pts_norm = centered / (radius + 1e-6)

    # subsample to fixed num_points
    M = pts_norm.shape[0]
    if M >= num_points:
        idx = np.random.choice(M, num_points, replace=False)
    else:
        pad = np.random.choice(M, num_points - M, replace=True)
        idx = np.concatenate([np.arange(M), pad], axis=0)
    pts_norm = pts_norm[idx]

    pts_norm_torch = torch.from_numpy(pts_norm).float()  # (N,3)
    return pts_norm_torch, center, radius, model


def compute_reward(
    coverage: float,
    num_used_views: int,
    num_views: int,
    coverage_target: float,
    lambda_views: float,
    alpha_shortfall: float,
) -> float:
    """
    Reward shaping:
        R = coverage
            - λ * (num_used / num_views)
            - α * max(0, coverage_target - coverage)^2
    """
    shortfall = max(0.0, coverage_target - coverage)
    reward = (
        coverage
        - lambda_views * (num_used_views / max(1, num_views))
        - alpha_shortfall * (shortfall ** 2)
    )
    return reward


# visioncraft_data.py

def evaluate_plan_on_model_single(
    planner,
    raw_actions: torch.Tensor,
    model,
    center: np.ndarray,
    radius: float,
    num_views: int,
    VisibilityManager,
    Viewpoint,
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
    fixed_radius_world: float = None,
    use_spherical_lookat: bool = False,
    use_gating: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate a single-shot plan (raw_actions) on a single Model instance.

    Modes:
    - default: use planner.decode_raw_actions (full SE(3)+radius+gate).
    - if fixed_radius_world is not None and use_spherical_lookat=True:
        * ignore planner.decode_raw_actions,
        * use only direction+gate from raw_actions to place cameras on a sphere
          of fixed_radius_world around 'center', and look-at 'center'.
    """
    device = next(planner.parameters()).device

    # raw_actions: (A,)
    raw_actions = raw_actions.to(device)
    per = planner.per_view_dim
    K = num_views

    if use_spherical_lookat and fixed_radius_world is not None:
        # -----------------------------------------
        # Manual decode: sphere radius + look-at center
        # -----------------------------------------
        # reshape to (K, per)
        x = raw_actions.view(K, per).detach().cpu().numpy()

        dir_raw = x[:, 0:3]    # (K,3)
        gate_raw = x[:, 7]     # (K,)

        # normalize directions, with fallback if norm is tiny
        norms = np.linalg.norm(dir_raw, axis=1, keepdims=True)
        eps = 1e-6
        mask_small = norms < eps
        norms_clamped = np.where(mask_small, 1.0, norms)  # avoid divide by zero
        dirs = dir_raw / norms_clamped

        # for any degenerate direction (nearly zero), set a default direction
        if mask_small.any():
            dirs[mask_small.squeeze(-1)] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # positions on fixed-radius sphere
        positions = center.reshape(1, 3) + fixed_radius_world * dirs  # (K,3)

        # gates in (0,1)
        gates = 1.0 / (1.0 + np.exp(-gate_raw))  # sigmoid

        ypr = None  # unused in this mode

    else:
        # -----------------------------------------
        # Default decode: use planner.decode_raw_actions
        # -----------------------------------------
        raw_actions_b = raw_actions.unsqueeze(0)  # (1,A)
        center_t = torch.from_numpy(center.astype(np.float32)).unsqueeze(0).to(device)

        r_min = r_min_factor * radius
        r_max = r_max_factor * radius

        positions_t, ypr_t, gates_t = planner.decode_raw_actions(
            raw_actions_b, r_min=r_min, r_max=r_max, center=center_t
        )

        positions = positions_t[0].detach().cpu().numpy()  # (K,3)
        ypr = ypr_t[0].detach().cpu().numpy()              # (K,3)
        gates = gates_t[0].detach().cpu().numpy()          # (K,)

    # -----------------------------------------
    # Build viewpoints and raycast
    # -----------------------------------------
    vm = VisibilityManager(model)

    used = 0
    gate_thresh = 0.5
    radii_list: List[float] = []

    for i in range(K):
        gate_val = gates[i]
        if use_gating and gate_val < gate_thresh:
            continue

        used += 1
        pos = positions[i]
        radii_list.append(float(np.linalg.norm(pos - center)))

        if use_spherical_lookat and fixed_radius_world is not None:
            # camera looks at object center
            vp = Viewpoint.from_lookat(
                pos.tolist(),
                center.tolist(),
            )
            vp.setNearPlane(near)
            vp.setFarPlane(far)
            vp.setDownsampleFactor(downsample)
        else:
            # original free-orientation mode
            yaw, pitch, roll = ypr[i]
            vec = np.array(
                [pos[0], pos[1], pos[2], yaw, pitch, roll],
                dtype=np.float64,
            )
            vp = Viewpoint.from_euler(
                vec,
                near,
                far,
                2448,
                2048,
                hfov,
                vfov,
            )
            vp.setDownsampleFactor(downsample)

        vm.trackViewpoint(vp)
        vp.performRaycastingOnGPU(model)

    coverage = float(vm.getCoverageScore())

    reward = compute_reward(
        coverage=coverage,
        num_used_views=used,
        num_views=K,
        coverage_target=coverage_target,
        lambda_views=lambda_views,
        alpha_shortfall=alpha_shortfall,
    )

    if len(radii_list) > 0:
        avg_radius = float(np.mean(radii_list))
    else:
        # fallback if no views used
        if fixed_radius_world is not None:
            avg_radius = fixed_radius_world
        else:
            r_min = r_min_factor * radius
            r_max = r_max_factor * radius
            avg_radius = float((r_min + r_max) * 0.5)

    metrics = {
        "coverage": coverage,
        "num_views_used": used,
        "avg_gate": float(gates.mean()),
        "avg_radius": avg_radius,
        "reward": reward,
    }
    return reward, metrics
