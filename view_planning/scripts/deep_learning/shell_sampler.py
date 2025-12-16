import numpy as np
import open3d as o3d

from gpu_sdf import fast_sdf


def main():
    # ----------------------------------------------------
    # CONFIG
    # ----------------------------------------------------
    mesh_path = "../../models/bracket.ply"

    inner_level = 200.0   # inner shell distance
    outer_level = 300.0  # outer shell distance
    num_candidates = 100000  # how many random points to sample
    max_shell_points = 100000 # max points to visualize from the shell

    # ----------------------------------------------------
    # LOAD MESH FOR BBOX + VISUALIZATION
    # ----------------------------------------------------
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    bbox = mesh.get_axis_aligned_bounding_box()
    minb = np.array(bbox.min_bound, dtype=np.float32)
    maxb = np.array(bbox.max_bound, dtype=np.float32)
    dims = maxb - minb

    # Pad box by outer_level so we can reach φ ≈ outer_level
    pad = outer_level * 1.2
    box_min = minb - pad
    box_max = maxb + pad

    print("Sampling candidates in box:")
    print("  min:", box_min)
    print("  max:", box_max)

    # ----------------------------------------------------
    # SAMPLE CANDIDATE POINTS
    # ----------------------------------------------------
    candidates = np.random.uniform(
        low=box_min,
        high=box_max,
        size=(num_candidates, 3)
    ).astype(np.float32)

    # ----------------------------------------------------
    # EVALUATE SDF
    # ----------------------------------------------------
    # fast_sdf caches evaluator from previous calls, so this
    # uses the same mesh / GPU data as your slice script.
    phi = fast_sdf(
        mesh_path,
        candidates,
        num_surface_points=3000,
        batch_grid=2048,
        batch_surf=2048,
    )

    # ----------------------------------------------------
    # SELECT POINTS IN THE SHELL [inner_level, outer_level]
    # ----------------------------------------------------
    # We only want the OUTSIDE band: φ in [inner_level, outer_level]
    shell_mask = (phi >= inner_level) & (phi <= outer_level)
    shell_points = candidates[shell_mask]

    print(f"Total candidates: {num_candidates}")
    print(f"Points in shell [{inner_level}, {outer_level}]: {shell_points.shape[0]}")

    if shell_points.shape[0] == 0:
        print("No shell points found with these levels. Adjust inner/outer_level.")
        return

    # Limit number of points for visualization
    if shell_points.shape[0] > max_shell_points:
        idx = np.random.choice(shell_points.shape[0], size=max_shell_points, replace=False)
        shell_points = shell_points[idx]

    # ----------------------------------------------------
    # BUILD OPEN3D POINT CLOUD FOR SHELL
    # ----------------------------------------------------
    shell_pcd = o3d.geometry.PointCloud()
    shell_pcd.points = o3d.utility.Vector3dVector(shell_points)
    shell_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # red shell

    mesh.paint_uniform_color([0.7, 0.7, 0.7])       # gray mesh

    # ----------------------------------------------------
    # VISUALIZE
    # ----------------------------------------------------
    print("Launching 3D visualization...")
    o3d.visualization.draw_geometries([mesh, shell_pcd])


if __name__ == "__main__":
    main()
