import numpy as np
import sys
import time
# Import visioncraft bindings
sys.path.append("../build/python_bindings")
from visioncraft_py import Model, Viewpoint, VisibilityManager, Visualizer

# Utility function to generate random positions on a sphere surface
def generate_random_positions(radius, count):
    return [
        np.array([
            radius * np.sin(phi) * np.cos(theta),
            radius * np.sin(phi) * np.sin(theta),
            radius * np.cos(phi)
        ])
        for theta, phi in zip(np.random.uniform(0, 2 * np.pi, count), 
                              np.random.uniform(0, np.pi, count))
    ]

# Initialize visualizer
visualizer = Visualizer()
visualizer.initializeWindow("3D View")
visualizer.setBackgroundColor([0.0, 0.0, 0.0])

# Load model and initialize VisibilityManager
model = Model()
model.loadModel("../models/gorilla.ply", 50000)
model.addVoxelProperty("alignment", 0.0)  # Initialize alignment property
visibility_manager = VisibilityManager(model)

# Calculate and set alignment scores based on a sample viewpoint
position = np.array([0.0, 0.0, 1800.0])  # Example viewpoint position
look_at = np.array([0.0, 0.0, 0.0])
viewpoint_z_axis = (look_at - position) / np.linalg.norm(look_at - position)

# Precompute voxel normals and alignment scores
voxel_normals = model.getVoxelNormals()
for key, voxel in model.getVoxelMap().items():
    if key in voxel_normals:
        voxel_normal = voxel_normals[key]
        alignment_value = max(0.0, -viewpoint_z_axis.dot(voxel_normal))
        model.setVoxelProperty(key, "alignment", alignment_value)

# Greedy algorithm to select viewpoints based on alignment-filtered coverage
target_coverage = 0.995
achieved_coverage = 0.0
selected_viewpoints = []
start_time = time.time()
alignment_threshold = 0.8  # Threshold for alignment filtering

# Greedy algorithm main loop
while achieved_coverage < target_coverage:
    best_viewpoint, best_coverage_increment = None, 0.0

    for position in generate_random_positions(400, 10):
        viewpoint = Viewpoint.from_lookat(position, [0.0, 0.0, 0.0])
        viewpoint.setNearPlane(300)
        viewpoint.setFarPlane(900.0)
        viewpoint.setDownsampleFactor(8.0)

        # Perform raycasting on GPU
        visibility_manager.trackViewpoint(viewpoint)
        visible_voxels = viewpoint.performRaycastingOnGPU(model)
       
        # Calculate filtered novel coverage for voxels with alignment > threshold
        novel_coverage = 0
        for key in visible_voxels:
            if model.getVoxelProperty(key, "alignment") > alignment_threshold:
                novel_coverage += 1

        # Update the best viewpoint if the current one has higher novel coverage
        if novel_coverage > best_coverage_increment:
            best_coverage_increment = novel_coverage
            best_viewpoint = viewpoint

        visibility_manager.untrackViewpoint(viewpoint)

    # Track the best viewpoint if found and update the achieved coverage
    if best_viewpoint:
        visibility_manager.trackViewpoint(best_viewpoint)
        best_viewpoint.performRaycastingOnGPU(model)

        # Update achieved coverage with the increment from the best viewpoint
        achieved_coverage += best_coverage_increment / len(model.getVoxelMap())
        selected_viewpoints.append(best_viewpoint)

# Final output
print(f"\nFinal Coverage Score: {achieved_coverage}")
print(f"Total Viewpoints Selected: {len(selected_viewpoints)}")
print(f"Time taken for greedy algorithm (excluding rendering): {time.time() - start_time:.2f} seconds")

# Visualization
visualizer.addVoxelMapProperty(model, "alignment", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
for viewpoint in selected_viewpoints:
    visualizer.addViewpoint(viewpoint, True, True)

visualizer.render()
