#!/usr/bin/env python3
"""
viewset_env_single_model.py

Single-object, single-shot view-set "environment" on a viewing sphere.

- Loads Visioncraft Model.
- Extracts a point cloud (for the policy input) with explicit debug.
- Provides functions to:
    * sample random view-sets on a sphere,
    * convert angles -> viewpoint positions,
    * evaluate coverage of any view-set using VisibilityManager + GPU raycasting,
    * run detailed debug prints (radii, per-view coverage, etc.).
"""

import os
import sys
import time
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

import numpy as np
import torch


@dataclass
class ViewSetEnvConfig:
    bindings_path: str
    model_path: str
    num_views: int = 6
    radius: float = 400.0      # sphere radius (world units)
    pc_size: int = 2048        # point cloud size for policy input
    near: float = 300.0
    far: float = 900.0
    downsample_factor: float = 2.0
    device: str = "cuda"


class ViewSetEnvSingleModel:
    """
    A lightweight "env" for single-shot view-set planning on a known object.

    - Object is fixed (single CAD model).
    - Viewpoints live on a sphere of radius R around the object center.
    - Each viewpoint looks at the object center.
    """

    def __init__(self, cfg: ViewSetEnvConfig):
        self.cfg = cfg

        # ------------------------------------------------------------------ #
        # 1) Import Visioncraft bindings
        # ------------------------------------------------------------------ #
        sys.path.append(cfg.bindings_path)
        try:
            from visioncraft_py import Model, Viewpoint, VisibilityManager, Visualizer
        except Exception as e:
            raise RuntimeError(
                f"Failed to import visioncraft_py from {cfg.bindings_path}: {e}"
            )

        self.Model = Model
        self.Viewpoint = Viewpoint
        self.VisibilityManager = VisibilityManager
        self.Visualizer = Visualizer

        # ------------------------------------------------------------------ #
        # 2) Load model once
        # ------------------------------------------------------------------ #
        self.model = Model()
        ok = self.model.loadModel(cfg.model_path, 250000)
        if not ok:
            raise RuntimeError(f"Model.loadModel({cfg.model_path}) returned False")

        print("Voxel normals computed and stored successfully.")  # from C++ side

        # Bounds / center sanity check
        self.min_bound = np.asarray(self.model.getMinBound(), dtype=np.float32)
        self.max_bound = np.asarray(self.model.getMaxBound(), dtype=np.float32)
        self.center_world = np.asarray(self.model.getCenter(), dtype=np.float32)

        print("\n[ENV] Model loaded")
        print(f"      min_bound = {self.min_bound}")
        print(f"      max_bound = {self.max_bound}")
        print(f"      center    = {self.center_world}")

        # ------------------------------------------------------------------ #
        # 3) Extract a point cloud for the policy input
        # ------------------------------------------------------------------ #
        self.pc_world = self._extract_point_cloud(cfg.pc_size)

        # Center on object center and normalize to unit ball for the net
        pc_centered = self.pc_world - self.center_world[None, :]
        radii = np.linalg.norm(pc_centered, axis=1)
        self.world_radius = float(radii.max())
        if self.world_radius < 1e-6:
            self.world_radius = 1.0

        self.pc_norm = (pc_centered / self.world_radius).astype(np.float32)

        print("\n[ENV] Point cloud debug")
        print(f"      pc_world shape      = {self.pc_world.shape}")
        print(f"      pc_world min        = {self.pc_world.min(axis=0)}")
        print(f"      pc_world max        = {self.pc_world.max(axis=0)}")
        print(f"      pc_world mean       = {self.pc_world.mean(axis=0)}")
        print(f"      pc_centered radius  = min={radii.min():.4f}, max={radii.max():.4f}")
        print(f"      Using world_radius  = {self.world_radius:.4f}")
        print(f"      pc_norm first[0:5]  =\n{self.pc_norm[:5]}")

        # Torch versions
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        if cfg.device == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available. Falling back to CPU.")

        self.pc_tensor = torch.from_numpy(self.pc_norm).to(self.device)

        # ------------------------------------------------------------------ #
        # 4) Basic voxel stats (for debugging)
        # ------------------------------------------------------------------ #
        voxel_keys = list(self.model.getVoxelMap())
        print(f"\n[ENV] Voxel map debug")
        print(f"      #voxels (surface shell) = {len(voxel_keys)}")
        if voxel_keys:
            print(f"      first key (OctoMap indices) = {voxel_keys[0]}")

        # Pre-create a Visualizer for later sanity checks (optional)
        self.visualizer = None

    # ---------------------------------------------------------------------- #
    # Internal: point cloud extraction
    # ---------------------------------------------------------------------- #
    def _extract_point_cloud(self, pc_size: int) -> np.ndarray:
        """
        Extract a point cloud from Visioncraft.

        There are several possible binding setups. We try a couple of
        reasonable options; if they don't exist, this will raise with
        a clear message so you can patch it.

        Adjust this function if your bindings are different.
        """
        # --- OPTION 1: direct point-cloud -> numpy binding -----------------
        # (You may already have something like this.)
        if hasattr(self.model, "getPointCloudAsNumpy"):
            pc = np.asarray(self.model.getPointCloudAsNumpy(), dtype=np.float32)
            print("[ENV] Using model.getPointCloudAsNumpy()")
        elif hasattr(self.model, "getPointCloud"):
            # Sometimes bindings expose .getPointCloud() that returns a list
            # of (x,y,z).
            raw = self.model.getPointCloud()
            pc = np.asarray(raw, dtype=np.float32)
            print("[ENV] Using model.getPointCloud()")
        else:
            # --- OPTION 2: build from MetaVoxel map positions -------------
            # Assumes there is some way to get per-voxel positions.
            print("[ENV] No direct point-cloud binding found; using voxel centers.")
            voxel_keys = list(self.model.getVoxelMap())
            if not voxel_keys:
                raise RuntimeError("model.getVoxelMap() returned empty; cannot build PC")

            positions = []
            # *** IMPORTANT ***
            # You likely have *some* binding to get MetaVoxel positions by key.
            # Common variants: model.getVoxelPosition(key), model.getMetaVoxelPosition(key),
            #                  model.getVoxel(key).getPosition()
            # Adjust this block to match your actual bindings.
            #
            # For now, we try model.getVoxel(key).getPosition() and log errors if it fails.
            if hasattr(self.model, "getVoxel"):
                print("[ENV] Using model.getVoxel(key).getPosition() to build PC")
                for k in voxel_keys:
                    try:
                        mv = self.model.getVoxel(k)
                        pos = np.array(mv.getPosition(), dtype=np.float32)
                        positions.append(pos)
                    except Exception as e:
                        raise RuntimeError(
                            "Failed to get voxel position from model.getVoxel(key). "
                            "Please adapt _extract_point_cloud() to your bindings."
                        ) from e
            else:
                raise RuntimeError(
                    "No getPointCloud* binding and model.getVoxel() "
                    "not found. Please expose a point-cloud or voxel-position "
                    "binding and adapt _extract_point_cloud()."
                )

            pc = np.stack(positions, axis=0)

        # Subsample / pad to pc_size for the policy
        if pc.shape[0] >= pc_size:
            idx = np.random.choice(pc.shape[0], size=pc_size, replace=False)
            pc = pc[idx]
        else:
            # repeat if there are fewer points than requested
            reps = int(np.ceil(pc_size / pc.shape[0]))
            pc = np.repeat(pc, reps, axis=0)[:pc_size]

        return pc

    # ---------------------------------------------------------------------- #
    # Angles <-> positions on sphere
    # ---------------------------------------------------------------------- #
    def angles_to_positions(self, angles: np.ndarray) -> np.ndarray:
        """
        Convert [num_views, 2] array of (theta, phi) to world positions
        on the viewing sphere.

        theta in [-pi, pi], phi in [0, pi]
        """
        assert angles.shape == (self.cfg.num_views, 2)
        theta = angles[:, 0]
        phi = angles[:, 1]

        # Directions on unit sphere
        dirs = np.stack(
            [
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi),
            ],
            axis=-1,
        )  # [K, 3]

        positions = self.center_world[None, :] + self.cfg.radius * dirs
        return positions.astype(np.float32)

    def sample_random_angles(self) -> np.ndarray:
        """
        Sample K random viewpoints on the sphere.
        Uniform theta in [0, 2pi), phi in [0, pi].
        """
        theta = np.random.uniform(0.0, 2.0 * math.pi, size=(self.cfg.num_views,))
        phi = np.arccos(1.0 - 2.0 * np.random.uniform(0.0, 1.0, size=(self.cfg.num_views,)))
        # phi ~ [0,π], uniformly over sphere
        return np.stack([theta, phi], axis=-1).astype(np.float32)

    # ---------------------------------------------------------------------- #
    # Coverage evaluation
    # ---------------------------------------------------------------------- #
    def evaluate_viewset(
        self,
        angles: np.ndarray,
        debug: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate coverage for a set of angles.

        Returns:
            coverage (float),
            info = {
                "positions": [K,3] positions in world coords,
                "coverage_per_step": [K] coverage after each added view,
                "viewpoints": list[Viewpoint],
            }
        """
        assert angles.shape == (self.cfg.num_views, 2)
        positions = self.angles_to_positions(angles)

        # Basic sanity on radii
        radii = np.linalg.norm(positions - self.center_world[None, :], axis=1)

        if debug:
            print("\n[ENV] evaluate_viewset() debug")
            print(f"      angles (first 3, deg) =")
            for i in range(min(3, len(angles))):
                print(
                    f"         view {i}: "
                    f"theta={angles[i,0]*180/math.pi:7.2f}°, "
                    f"phi={angles[i,1]*180/math.pi:7.2f}°"
                )
            print(f"      positions (first 3)   =\n{positions[:3]}")
            print(f"      radii min/max/mean    = {radii.min():.3f}, "
                  f"{radii.max():.3f}, {radii.mean():.3f}")

        # Create a fresh VisibilityManager for this evaluation
        vm = self.VisibilityManager(self.model)

        viewpoints: List[Any] = []
        coverage_per_step: List[float] = []

        for i, pos in enumerate(positions):
            vp = self.Viewpoint.from_lookat(
                pos.tolist(),
                self.center_world.tolist(),
            )
            vp.setNearPlane(self.cfg.near)
            vp.setFarPlane(self.cfg.far)
            vp.setDownsampleFactor(self.cfg.downsample_factor)

            vm.trackViewpoint(vp)
            vp.performRaycastingOnGPU(self.model)

            cov = vm.getCoverageScore()
            coverage_per_step.append(cov)
            viewpoints.append(vp)

            if debug:
                print(
                    f"      step {i:02d}: "
                    f"coverage={cov:.4f} "
                    f"pos={pos}"
                )

        coverage = coverage_per_step[-1] if coverage_per_step else 0.0
        info = {
            "positions": positions,
            "coverage_per_step": np.array(coverage_per_step, dtype=np.float32),
            "viewpoints": viewpoints,
        }
        return coverage, info

    def evaluate_random_viewset(
        self,
        num_trials: int = 4,
        debug: bool = False,
    ) -> Tuple[float, float]:
        """
        Evaluate a few random view-sets to estimate the random baseline.

        Returns:
            (mean_random_cov, best_random_cov)
        """
        covs = []
        for t in range(num_trials):
            angles = self.sample_random_angles()
            cov, _ = self.evaluate_viewset(angles, debug=False)
            covs.append(cov)
        covs = np.array(covs, dtype=np.float32)
        if debug:
            print(f"[ENV] Random baseline over {num_trials} trials: "
                  f"mean={covs.mean():.4f}, max={covs.max():.4f}")
        return float(covs.mean()), float(covs.max())

    # ---------------------------------------------------------------------- #
    # Visualization helper (optional)
    # ---------------------------------------------------------------------- #
    def visualize_viewset(self, viewpoints: List[Any]):
        """
        Show the selected viewpoints + visibility in the Visioncraft Visualizer.
        """
        if self.visualizer is None:
            self.visualizer = self.Visualizer()
            self.visualizer.initializeWindow("View-set Visualization")
            self.visualizer.setBackgroundColor([0.0, 0.0, 0.0])
            # show voxels with visibility property (white→green)
            self.visualizer.addVoxelMapProperty(
                self.model,
                "visibility",
                [1.0, 1.0, 1.0],   # base color
                [0.0, 1.0, 0.0],   # highlight
            )

        for vp in viewpoints:
            # Show frustum and axes
            self.visualizer.addViewpoint(vp, True, True)

        print("Press Ctrl+C in the terminal to exit the viewer.")
        try:
            import time as _time
            while True:
                self.visualizer.renderStep()
                _time.sleep(0.01)
        except KeyboardInterrupt:
            print("[VIS] Viewer closed by user.")
