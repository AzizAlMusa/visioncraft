import open3d as o3d
import numpy as np
import octomap

def create_octree_from_point_cloud(pcd, resolution):
    # Create an empty octree with the specified resolution
    tree = octomap.OcTree(resolution)
    
    # Add points from the point cloud to the octree
    for point in np.asarray(pcd.points):
        tree.updateNode(point, True)
    
    # Update inner occupancy values
    tree.updateInnerOccupancy()
    return tree

def save_octree_to_files(tree, grid_size, output_prefix):
    voxel_array = np.zeros((grid_size, grid_size, grid_size))
    with open(f"{output_prefix}_octree.txt", "w") as file:
        for node in tree:
            if tree.isNodeOccupied(node):
                x, y, z = node.getCoordinate()
                prob = node.getOccupancy()
                voxel_index = (int(x), int(y), int(z))
                if all(0 <= idx < grid_size for idx in voxel_index):
                    voxel_array[voxel_index] = prob
                    file.write(f"{voxel_index[0]} {voxel_index[1]} {voxel_index[2]} {prob:.6f}\n")
    
    # Save the numpy array
    np.save(f"{output_prefix}_octomap.npy", voxel_array)

def process_mesh_file(file_path, file_type, resolution, grid_size, output_prefix):
    # Load the mesh file
    if file_type == 'stl':
        mesh = o3d.io.read_triangle_mesh(file_path)
        pcd = mesh.sample_points_uniformly(number_of_points=100000)
    elif file_type == 'ply':
        pcd = o3d.io.read_point_cloud(file_path)
    else:
        raise ValueError("Unsupported file type. Use 'stl' or 'ply'.")

    # Create octree from point cloud
    tree = create_octree_from_point_cloud(pcd, resolution)
    # Save octree to files
    save_octree_to_files(tree, grid_size, output_prefix)

# Example usage
file_path = "./data/stanford_bunny.ply"  # Replace with your STL or PLY file path
file_type = "ply"  # Set to 'stl' or 'ply' based on your file type
resolution = 0.01  # Set the resolution of the octree
grid_size = 32  # Define the size of the grid
output_prefix = "bunny"
process_mesh_file(file_path, file_type, resolution, grid_size, output_prefix)
