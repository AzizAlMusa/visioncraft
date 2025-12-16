# viewset_env.py
#
# Environment for one-shot viewset selection on a sphere using Visioncraft.
# Supports a variable number of viewpoints per evaluation (K can change),
# even though a nominal num_views is stored for configuration/debug.

import numpy as np
import torch


class ViewSetEnv(object):
    """
    One-shot viewset environment:
      - Known CAD model (via Visioncraft Model)
      - A set of K directions on the unit sphere
      - Directions mapped to viewpoints on a fixed-radius sphere around the object
      - Each viewpoint looks at model center
      - Reward = coverage (0..1) from a fresh VisibilityManager for *this* viewset only
    """

    def __init__(self,
                 model_path,
                 Model,
                 Viewpoint,
                 VisibilityManager,
                 num_views=4,
                 radius=400.0,
                 pc_size=2048,
                 device="cpu",
                 debug=True):
        """
        model_path        : path to .ply model
        Model, Viewpoint,
        VisibilityManager : classes imported from visioncraft_py
        num_views         : nominal max number of viewpoints (used for config/debug)
        radius            : sphere radius (world units)
        pc_size           : number of points for point cloud input
        device            : 'cpu' or 'cuda'
        debug             : if True, prints debug info during init and first eval
        """
        self.model_path = model_path
        self.Model = Model
        self.Viewpoint = Viewpoint
        self.VisibilityManager = VisibilityManager
        self.num_views = num_views
        self.radius = float(radius)
        self.pc_size = int(pc_size)
        self.device = torch.device(device)
        self.debug = debug

        # --- Load model once ---
        self.model = Model()
        ok = self.model.loadModel(self.model_path, 250000)
        if not ok:
            raise RuntimeError("Model.loadModel('{}') failed".format(self.model_path))

        # Bounds / center for info
        self.min_bound = np.array(self.model.getMinBound(), dtype=np.float32)
        self.max_bound = np.array(self.model.getMaxBound(), dtype=np.float32)
        self.center = np.array(self.model.getCenter(), dtype=np.float32)

        # Voxel info just for debug
        try:
            voxel_map = self.model.getVoxelMap()
            self.num_voxels = len(voxel_map)
        except Exception:
            self.num_voxels = -1  # unknown

        # --- Get point cloud as features ---
        # This uses your C++ binding: Model.get_point_cloud_array()
        raw_pc = self.model.get_point_cloud_array()  # (N, 3) float
        raw_pc = np.asarray(raw_pc, dtype=np.float32)
        if raw_pc.ndim != 2 or raw_pc.shape[1] != 3:
            raise RuntimeError("Expected point cloud of shape (N,3), got {}".format(raw_pc.shape))

        # center and normalize radius
        pc_centered = raw_pc - self.center[None, :]
        radii = np.linalg.norm(pc_centered, axis=1)
        max_r = float(radii.max())
        self.pc_radius_world = max_r
        self.pc_norm = pc_centered / (max_r + 1e-8)

        # subsample to pc_size
        n_points = self.pc_norm.shape[0]
        if n_points >= self.pc_size:
            idx = np.random.choice(n_points, size=self.pc_size, replace=False)
        else:
            idx = np.random.choice(n_points, size=self.pc_size, replace=True)
        self.pc_norm = self.pc_norm[idx, :]  # (pc_size, 3)

        # convert to torch once
        self.pc_tensor = torch.from_numpy(self.pc_norm).to(self.device)

        if self.debug:
            print("[ENV] Model loaded")
            print("[ENV] Bounds: min={}, max={}, center={}".format(
                self.min_bound, self.max_bound, self.center))
            print("[ENV] Using get_point_cloud_array() for PC")
            print("[ENV] Raw PC size: {}".format(raw_pc.shape))
            print("[ENV] pc_radius_world ≈ {:.4f}".format(self.pc_radius_world))
            print("[ENV] pc_norm min={}, max={}".format(self.pc_norm.min(axis=0),
                                                        self.pc_norm.max(axis=0)))
            print("[ENV] #voxels = {}".format(self.num_voxels))

    # ---------- Public API ----------

    def get_state(self):
        """Return normalized point cloud tensor on the correct device."""
        return self.pc_tensor

    def evaluate_viewset(self, dirs_tensor, debug=False, return_viewpoints=False):
        """
        Evaluate coverage for a set of directions.

        dirs_tensor: torch.Tensor or np.ndarray of shape (M, 3)
                     each row is an arbitrary 3D vector; we renormalize to unit.

        Returns:
            coverage (float) in [0,1]
            optionally also list[Viewpoint] if return_viewpoints=True
        """
        # Move to CPU & numpy for Visioncraft
        if isinstance(dirs_tensor, torch.Tensor):
            dirs = dirs_tensor.detach().cpu().numpy()
        else:
            dirs = np.asarray(dirs_tensor, dtype=np.float32)

        if dirs.ndim != 2 or dirs.shape[1] != 3:
            raise ValueError("Expected dirs shape (M,3), got {}".format(dirs.shape))

        num_views = dirs.shape[0]
        if num_views == 0:
            raise ValueError("evaluate_viewset received zero directions (M=0).")

        # Normalize directions
        norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
        dirs_unit = dirs / norms

        if debug or self.debug:
            print("[ENV][DBG] directions norms: {}".format(
                np.round(np.linalg.norm(dirs_unit, axis=1), 6)))

        # Create a fresh VisibilityManager for THIS evaluation
        vm = self.VisibilityManager(self.model)
        # ensure coverage starts at zero
        _ = vm.getCoverageScore()

        viewpoints = []

        for i in range(num_views):
            direction = dirs_unit[i]
            # viewpoint position on sphere
            pos_world = self.center + direction * self.radius

            vp = self.Viewpoint.from_lookat(
                pos_world.astype(np.float32),
                self.center.astype(np.float32)
            )
            # Use your usual camera settings
            vp.setNearPlane(300.0)
            vp.setFarPlane(900.0)
            vp.setDownsampleFactor(2.0)

            vm.trackViewpoint(vp)
            vp.performRaycastingOnGPU(self.model)

            viewpoints.append(vp)

            if (debug or self.debug) and i == 0:
                print("[ENV][DBG] View 0 pos (world) = {}".format(pos_world))

        coverage = vm.getCoverageScore()

        if debug or self.debug:
            try:
                vis_count = vm.getVisibilityCount()
                num_vis = len(vis_count)
            except Exception:
                num_vis = -1
            print("[ENV][DBG] coverage after {} views = {:.4f}, #visible_voxels = {}".format(
                num_views, coverage, num_vis))

        # Optional clean-up of observer links (break cycles)
        try:
            vm.untrackAllViewpoints()
        except Exception:
            pass

        if return_viewpoints:
            return float(coverage), viewpoints
        else:
            return float(coverage)
