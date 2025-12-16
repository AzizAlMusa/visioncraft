import numpy as np
import open3d as o3d
import cupy as cp
import time

# ---------------------------------------------------------------------
# INTERNAL CACHE SO MESH IS NOT RELOADED EACH CALL
# ---------------------------------------------------------------------
_EVALUATORS = {}


class _GPUSDFEvaluator:
    """
    Internal class. Do NOT call directly.
    Use fast_sdf(mesh_path, points) instead.
    """

    def __init__(
        self,
        mesh_path,
        num_surface_points=3000,
        batch_grid=2048,
        batch_surf=2048,
    ):
        self.mesh_path = mesh_path
        self.num_surface_points = num_surface_points
        self.batch_grid = batch_grid
        self.batch_surf = batch_surf

        # Load mesh
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
        self.mesh = mesh

        bbox = mesh.get_axis_aligned_bounding_box()
        self.bbox_min = np.array(bbox.min_bound, dtype=np.float32)
        self.bbox_max = np.array(bbox.max_bound, dtype=np.float32)

        # Surface Poisson sampling
        surf = mesh.sample_points_poisson_disk(self.num_surface_points)
        surf_pts = np.asarray(surf.points, dtype=np.float32)
        surf_nrm = np.asarray(surf.normals, dtype=np.float32)

        # Upload to GPU (float16)
        self.surf_pts_gpu = cp.asarray(surf_pts, dtype=cp.float16)
        self.surf_nrm_gpu = cp.asarray(surf_nrm, dtype=cp.float16)
        self.num_surf = self.surf_pts_gpu.shape[0]

    # ------------------------------------------------------------
    # The core SDF evaluator
    # ------------------------------------------------------------
    def evaluate(self, points_np):
        points_np = points_np.astype(np.float32)
        N = points_np.shape[0]

        unsigned = np.zeros(N, dtype=np.float32)
        sign = np.ones(N, dtype=np.float32)

        num_grid_batches = (N + self.batch_grid - 1) // self.batch_grid
        num_surf_tiles = (self.num_surf + self.batch_surf - 1) // self.batch_surf

        for bi in range(num_grid_batches):
            g0 = bi * self.batch_grid
            g1 = min(g0 + self.batch_grid, N)

            batch = cp.asarray(points_np[g0:g1], dtype=cp.float16)

            # running minima
            min_d2 = cp.full((batch.shape[0],), 1e20, dtype=cp.float32)
            min_idx = cp.zeros((batch.shape[0],), dtype=cp.int32)

            # tile surface pts
            for ti in range(num_surf_tiles):
                s0 = ti * self.batch_surf
                s1 = min(s0 + self.batch_surf, self.num_surf)

                surf_tile = self.surf_pts_gpu[s0:s1]

                diff = batch[:, None, :] - surf_tile[None, :, :]  
                d2 = cp.sum(diff * diff, axis=2, dtype=cp.float32)

                local_idx = cp.argmin(d2, axis=1)
                local_min_d2 = d2[cp.arange(d2.shape[0]), local_idx]

                mask = local_min_d2 < min_d2
                min_d2 = cp.where(mask, local_min_d2, min_d2)
                min_idx = cp.where(mask, local_idx + s0, min_idx)

            # unsigned distance
            unsigned[g0:g1] = cp.sqrt(min_d2).get()

            # sign from surface normals
            closest_pts = self.surf_pts_gpu[min_idx]
            closest_nrm = self.surf_nrm_gpu[min_idx]
            vec = batch - closest_pts
            dot = cp.sum(vec * closest_nrm, axis=1)
            sign[g0:g1] = cp.where(dot < 0, -1.0, 1.0).get()

        return unsigned * sign


# ---------------------------------------------------------------------
# PUBLIC INTERFACE
# ---------------------------------------------------------------------

def fast_sdf(
    mesh_path,
    query_points,
    num_surface_points=3000,
    batch_grid=2048,
    batch_surf=2048,
):
    """
    Compute the signed distance φ(x) for arbitrary 3D points.

    This function automatically:
    - caches the mesh + GPU upload on first call
    - reuses them on repeated calls
    - returns SDF for the given query_points
    
    Parameters
    ----------
    mesh_path : str
        Path to mesh file ('../../models/gorilla.ply').
    query_points : (N,3) np.ndarray
        Arbitrary 3D points to evaluate SDF at.
    num_surface_points : int
        Number of Poisson disk samples used to approximate the surface.
        Smaller = faster but less accurate.
        Typical: 2000–5000.
    batch_grid : int
        Number of points processed per GPU batch.
        Larger = faster but uses more GPU RAM.
        Safe for 1050Ti: 1024–4096.
    batch_surf : int
        Tile size for surface points during nearest-search.
        Larger = faster but uses more GPU RAM.
        Safe for 1050Ti: 1024–4096.

    Returns
    -------
    sdf : (N,) np.ndarray
        Signed distance at the query points.
        Positive outside, negative inside.
    """

    # Build or reuse evaluator
    key = (mesh_path, num_surface_points, batch_grid, batch_surf)
    
    if key not in _EVALUATORS:
        _EVALUATORS[key] = _GPUSDFEvaluator(
            mesh_path,
            num_surface_points=num_surface_points,
            batch_grid=batch_grid,
            batch_surf=batch_surf,
        )

    evaluator = _EVALUATORS[key]
    return evaluator.evaluate(query_points)
