# nbv_env_discrete.py

import sys
import math
import numpy as np
from typing import Optional, List


class NBVDiscreteEnv:
    """
    Simple discrete NBV environment using a precomputed candidate-view visibility matrix.

    - vis_matrix: (N_views, N_voxels) bool array
      vis_matrix[i, v] = True iff candidate view i sees voxel v.

    - State:
        covered_mask: (N_voxels,) bool
        used_mask:    (N_views,)  bool
        step_count:   int

      Observation is a flat float32 vector:
        obs = concat( used_mask, [coverage, step_fraction] )

    - Action: integer in [0, N_views-1]
    - Reward at step t: novel coverage fraction contributed by that view.
    - Episode ends when:
        * step_count >= max_steps, or
        * coverage >= coverage_target (if set).
    """

    def __init__(self, vis_matrix: np.ndarray,
                 max_steps: int,
                 coverage_target: Optional[float] = None):
        assert vis_matrix.ndim == 2
        self.vis_matrix = vis_matrix.astype(bool)
        self.num_views, self.num_voxels = self.vis_matrix.shape
        self.max_steps = max_steps
        self.coverage_target = coverage_target

        # Internal state
        self.covered_mask = None
        self.used_mask = None
        self.step_count = 0

    @property
    def obs_dim(self) -> int:
        return self.num_views + 2  # used_mask + coverage + step_fraction

    def reset(self):
        self.covered_mask = np.zeros(self.num_voxels, dtype=bool)
        self.used_mask = np.zeros(self.num_views, dtype=bool)
        self.step_count = 0
        return self._make_obs()

    def _make_obs(self):
        coverage = float(self.covered_mask.mean())
        step_frac = float(self.step_count / max(1, self.max_steps))
        obs = np.concatenate(
            [self.used_mask.astype(np.float32),
             np.array([coverage, step_frac], dtype=np.float32)],
            axis=0,
        )
        return obs.astype(np.float32)

    def step(self, action: int):
        assert 0 <= action < self.num_views
        self.step_count += 1

        if self.used_mask[action]:
            # Re-selecting a used view gives zero reward
            reward = 0.0
        else:
            cand_vis = self.vis_matrix[action]               # (N_voxels,)
            novel = np.logical_and(cand_vis, ~self.covered_mask)
            novel_count = int(novel.sum())
            reward = novel_count / float(self.num_voxels)

            self.covered_mask[novel] = True
            self.used_mask[action] = True

        coverage = float(self.covered_mask.mean())

        done = False
        if self.step_count >= self.max_steps:
            done = True
        if self.coverage_target is not None and coverage >= self.coverage_target:
            done = True

        obs = self._make_obs()
        info = {
            "coverage": coverage,
            "step": self.step_count,
        }
        return obs, reward, done, info


# --------------------------------------------------------------------------
#  Visioncraft glue: build vis_matrix from Model / Viewpoint / VisibilityManager
# --------------------------------------------------------------------------


def fibonacci_sphere(num_points: int, radius: float, center: np.ndarray) -> List[np.ndarray]:
    """
    Generate approximately uniform points on a sphere using Fibonacci sampling.
    All points are placed at 'radius' distance from 'center'.
    """
    points: List[np.ndarray] = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(num_points):
        y = 1.0 - (2.0 * (i + 0.5) / num_points)
        r = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * i

        x = math.cos(theta) * r
        z = math.sin(theta) * r
        pos = center + radius * np.array([x, y, z], dtype=np.float32)
        points.append(pos)
    return points


def build_env_from_visioncraft(bindings_path: str,
                               model_path: str,
                               num_candidates: int = 64,
                               max_steps: int = 6,
                               fixed_radius: float = 400.0,
                               downsample_factor: float = 4.0,
                               seed: int = 0,
                               verbose: bool = True) -> NBVDiscreteEnv:
    """
    1) Load model via visioncraft_py.Model (calls generateAllStructures internally).
    2) Extract all voxels from MetaVoxelMap via Model.getVoxelMap().
    3) Sample 'num_candidates' viewpoints on a sphere of radius fixed_radius around model center.
    4) For each candidate, run one GPU raycast and get visible voxels from VisibilityManager.
    5) Build vis_matrix and construct NBVDiscreteEnv.
    """

    import importlib

    np.random.seed(seed)

    sys.path.append(bindings_path)
    vc_mod = importlib.import_module("visioncraft_py")
    Model = getattr(vc_mod, "Model")
    Viewpoint = getattr(vc_mod, "Viewpoint")
    VisibilityManager = getattr(vc_mod, "VisibilityManager")

    model = Model()
    ok = model.loadModel(model_path, 250000)
    if not ok:
        raise RuntimeError(f"Model.loadModel failed for {model_path}")

    # Get voxel keys from MetaVoxelMap via getVoxelMap()
    voxel_list = list(model.getVoxelMap())
    if len(voxel_list) == 0:
        raise RuntimeError("Model.getVoxelMap() returned no voxels")

    num_voxels = len(voxel_list)
    voxel_index = {v: i for i, v in enumerate(voxel_list)}

    if verbose:
        print(f"[ENV] #voxels = {num_voxels}")

    # Camera sphere center & radius from mesh bounds
    min_b = np.asarray(model.getMinBound(), dtype=np.float32)
    max_b = np.asarray(model.getMaxBound(), dtype=np.float32)
    center = np.asarray(model.getCenter(), dtype=np.float32)
    radius_world = 0.5 * float(np.linalg.norm(max_b - min_b))

    if verbose:
        print(f"[ENV] world radius ~ {radius_world:.2f}, center = {center}")

    cam_radius = fixed_radius  # for now, keep explicit

    # Sample candidate positions
    cand_positions = fibonacci_sphere(num_candidates, cam_radius, center)

    # Build visibility matrix
    vis_matrix = np.zeros((num_candidates, num_voxels), dtype=bool)

    vm = VisibilityManager(model)

    for i, pos in enumerate(cand_positions):
        if verbose and (i % 10 == 0 or i == num_candidates - 1):
            print(f"[ENV] Candidate {i+1}/{num_candidates}")

        vp = Viewpoint.from_lookat(pos.tolist(), center.tolist())
        # reuse your typical near/far/downsample
        vp.setNearPlane(300.0)
        vp.setFarPlane(900.0)
        vp.setDownsampleFactor(downsample_factor)

        vm.trackViewpoint(vp)
        vp.performRaycastingOnGPU(model)

        vmap = vm.getVisibilityMap()  # dict: vp -> set of voxel keys
        voxels = vmap.get(vp, None)
        if voxels is None:
            # Fallback: iterate and find matching vp
            for k, vs in vmap.items():
                if k is vp:
                    voxels = vs
                    break

        if voxels is None:
            continue

        for v in voxels:
            idx = voxel_index.get(v, None)
            if idx is not None:
                vis_matrix[i, idx] = True

    if verbose:
        # quick sanity: average per-view coverage
        per_view_cov = vis_matrix.sum(axis=1) / float(num_voxels)
        print(f"[ENV] mean single-view coverage: {per_view_cov.mean():.4f}, "
              f"max: {per_view_cov.max():.4f}")

    env = NBVDiscreteEnv(vis_matrix=vis_matrix,
                         max_steps=max_steps,
                         coverage_target=None)
    # Attach num_views convenience attr used by training script
    env.num_views = vis_matrix.shape[0]
    return env
