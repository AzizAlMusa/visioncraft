import open3d as o3d

# Load the 3D mesh
mesh = o3d.io.read_triangle_mesh("../models/cat_experiment.ply")
mesh.compute_vertex_normals()

# Create coordinate frame at origin (default is size=1.0, origin=[0,0,0])
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0, origin=[0, 0, 0])

# Visualize mesh and axes
o3d.visualization.draw_geometries([mesh, axes])
