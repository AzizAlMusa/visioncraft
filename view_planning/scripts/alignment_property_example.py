import numpy as np
import sys
import os

# Adjust path to locate bindings if necessary
sys.path.append(os.path.abspath("../build/python_bindings"))
from visioncraft_py import Model, Visualizer, Viewpoint

# Initialize the Visualizer
visualizer = Visualizer()
visualizer.initializeWindow("3D View")
visualizer.setBackgroundColor(np.array([0.0, 0.0, 0.0]))

# Load model
model = Model()
model.loadModel("../models/wavy_cube.ply", 1000000)
model.addVoxelProperty("alignment", 0.0)  # Initialize alignment property for all voxels
model.addVoxelProperty("normal", np.array([0.0, 0.0, 1.0]))  # Default normal property for each voxel

# Set up the viewpoint
position = np.array([0.0, 0.0, 1800.0])  # Example viewpoint position
look_at = np.array([0.0, 0.0, 0.0])    # Look at the origin
viewpoint = Viewpoint.from_lookat(position, look_at)
viewpoint.setDownsampleFactor(8.0)
viewpoint.setNearPlane(100)
viewpoint.setFarPlane(2000)

# Calculate the z-axis direction of the viewpoint
viewpoint_z_axis = (look_at - position) / np.linalg.norm(look_at - position)

# Retrieve voxel normals once to avoid repeated calls
voxel_normals = model.getVoxelNormals()  # Dictionary with keys and corresponding normals

# Iterate through each key in the meta voxel map
for key, voxel in model.getVoxelMap().items():
    # Check if the normal exists for the given key in voxel_normals
    if key in voxel_normals:
        voxel_normal = voxel_normals[key]  # Retrieve normal from the precomputed normals
        model.setVoxelProperty(key, "normal", voxel_normal)  # Set the normal property for the voxel

        # Calculate alignment as the dot product between the viewpoint's z-axis and the voxel normal
        alignment_value = max(0.0, -viewpoint_z_axis.dot(voxel_normal))  # Ensure value is between 0 and 1
        model.setVoxelProperty(key, "alignment", alignment_value)  # Set the alignment property
    else:
        print(f"Warning: No normal found for voxel key {key}")

# Visualize the 'alignment' property
baseColor = np.array([0.0, 0.0, 1.0])      # Base color for low alignment
propertyColor = np.array([1.0, 0.0, 0.0])  # Color for high alignment
visualizer.addVoxelMapProperty(model, "alignment", baseColor, propertyColor)
visualizer.addViewpoint(viewpoint, True, True)
# Start rendering loop
visualizer.render()
