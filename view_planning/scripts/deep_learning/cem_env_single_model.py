# cem_env_single_model.py

import numpy as np


def build_normalized_pointcloud_from_voxels(model, num_points: int = 2048):
    """
    Build a normalized point cloud from the model's voxel map.

    We only need a *shape* descriptor for the network, not exact metric units.
    So we:
      - grab all voxel "coordinates" from model.getVoxelMap() (whatever
        the binding returns for the MetaVoxelMap keys/positions),
      - treat them as 3D points in some coordinate system,
      - normalize to zero mean and unit-ish radius.

    This does NOT affect viewpoint placement, which uses model.getCenter()
    and the mesh bounds directly (world coordinates). :contentReference[oaicite:5]{index=5}
    """
    voxel_iter = model.getVoxelMap()
    voxels = list(voxel_iter)
    if len(voxels) == 0:
        raise RuntimeError("Model.getVoxelMap() returned no voxels")

    pts = np.asarray(voxels, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise RuntimeError(f"Expected voxel map as (N,3), got {pts.shape}")

    # Subsample if needed
    if pts.shape[0] > num_points:
        idx = np.random.choice(pts.shape[0], size=num_points, replace=False)
        pts = pts[idx]

    # Normalize in this coordinate space (voxel key or world coords, doesn't matter)
    center_pc = pts.mean(axis=0, keepdims=True)
    extents = pts.max(axis=0) - pts.min(axis=0)
    radius_pc = float(np.linalg.norm(extents) * 0.5 + 1e-6)

    pts_norm = (pts - center_pc) / radius_pc

    return pts_norm.astype(np.float32)


def get_world_center_and_radius(model):
    """
    Use the mesh AABB to get world-space center and 'radius'.
    This is used ONLY for placing cameras in world coordinates. :contentReference[oaicite:6]{index=6}
    """
    min_b = np.asarray(model.getMinBound(), dtype=np.float32)
    max_b = np.asarray(model.getMaxBound(), dtype=np.float32)
    center = np.asarray(model.getCenter(), dtype=np.float32)

    radius = float(0.5 * np.linalg.norm(max_b - min_b))
    return center, radius


def decode_spherical_actions_to_directions(raw_actions: np.ndarray) -> np.ndarray:
    """
    raw_actions: (K,2) array: [theta_raw, phi_raw]
    Returns: dirs: (K,3) unit vectors.

    Mapping:
      theta_raw -> theta ∈ [0, 2π)
      phi_raw   -> phi   ∈ [0, π]
    """
    if raw_actions.ndim != 2 or raw_actions.shape[1] != 2:
        raise ValueError(f"Expected raw_actions shape (K,2), got {raw_actions.shape}")

    theta_raw = raw_actions[:, 0]
    phi_raw = raw_actions[:, 1]

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    theta = 2.0 * np.pi * sigmoid(theta_raw)     # [0, 2π)
    phi = np.pi * sigmoid(phi_raw)               # [0, π]

    sin_phi = np.sin(phi)
    x = sin_phi * np.cos(theta)
    y = sin_phi * np.sin(theta)
    z = np.cos(phi)

    dirs = np.stack([x, y, z], axis=-1)  # (K,3)
    return dirs.astype(np.float32)


def evaluate_sphere_plan_single_model(
    raw_actions: np.ndarray,
    model,
    center_world: np.ndarray,
    radius_world: float,
    num_views: int,
    VisibilityManager,
    Viewpoint,
    fixed_radius: float = 400.0,
    near: float = 300.0,
    far: float = 900.0,
    downsample: float = 4.0,
    hfov: float = 44.8,
    vfov: float = 42.6,
) -> float:
    """
    Evaluate a K-view plan on a single model, with cameras constrained to a
    sphere of radius 'fixed_radius' around 'center_world', always looking at
    the center.

    raw_actions: (K,2) numpy array of raw spherical angles.
    Returns: coverage in [0,1].
    """
    K, D = raw_actions.shape
    assert K == num_views, f"raw_actions K={K} != num_views={num_views}"
    assert D == 2

    dirs = decode_spherical_actions_to_directions(raw_actions)  # (K,3)

    vm = VisibilityManager(model)

    for k in range(K):
        pos = center_world + fixed_radius * dirs[k]

        vp = Viewpoint.from_lookat(
            pos.tolist(),
            center_world.tolist(),
        )
        vp.setNearPlane(near)
        vp.setFarPlane(far)
        vp.setDownsampleFactor(downsample)

        vm.trackViewpoint(vp)
        vp.performRaycastingOnGPU(model)

    coverage = float(vm.getCoverageScore())
    return coverage
