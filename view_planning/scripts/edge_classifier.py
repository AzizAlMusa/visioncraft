import open3d as o3d
import numpy as np

# === 1. Load the mesh ===
mesh = o3d.io.read_triangle_mesh("../models/mug.ply")
mesh.compute_vertex_normals()
vertices = np.asarray(mesh.vertices)

# === 2. Create scene and add mesh ===
scene = o3d.t.geometry.RaycastingScene()
mesh_o3d_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
_ = scene.add_triangles(mesh_o3d_t)

# === 3. Choose camera/viewpoint (adjust as needed) ===
# Example: camera located at +Z axis, looking toward origin
camera_origin = np.array([0.0, 0.0, 1.0]) * (np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
directions = vertices - camera_origin
directions /= np.linalg.norm(directions, axis=1, keepdims=True)

# === 4. Raycast to each vertex ===
rays = np.zeros((len(vertices), 6), dtype=np.float32)
rays[:, :3] = camera_origin
rays[:, 3:] = directions
rays_o3d = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

# === 5. Perform raycasting ===
ans = scene.cast_rays(rays_o3d)
t_hits = ans['t_hit'].numpy()

# A vertex is visible if the first hit is closer than the vertex itself
vertex_dists = np.linalg.norm(vertices - camera_origin, axis=1)
is_occluded = (t_hits < vertex_dists - 1e-4)  # Allow small epsilon

# === 6. Assign colors ===
colors = np.zeros((len(vertices), 3))
colors[is_occluded] = [1, 0, 0]    # Red
colors[~is_occluded] = [0, 1, 0]   # Green

# === 7. Visualize result ===
mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([mesh])
