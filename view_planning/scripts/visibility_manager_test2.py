import numpy as np
import sys
import os

# Adjust path to locate bindings if necessary
sys.path.append(os.path.abspath("../build/python_bindings"))
from visioncraft_py import Model, Viewpoint, VisibilityManager, Visualizer

# Initialize the Visualizer and set background color
visualizer = Visualizer()
visualizer.initializeWindow("3D View")
visualizer.setBackgroundColor(np.array([0.0, 0.0, 0.0]))

# Create a model instance and load a model file
model = Model()
model.loadModel("../models/cube.ply", 50000)

# Create a VisibilityManager for the model
visibility_manager = VisibilityManager(model)

# Define positions for six viewpoints
positions = [
    np.array([400.0, 0.0, 0.0]),
    np.array([-400.0, 0.0, 0.0]),
    np.array([0.0, 400.0, 0.0]),
    np.array([0.0, -400.0, 0.0]),
    np.array([0.0, 0.0, 400.0]),
    np.array([0.0, 0.0, -400.0]),
    np.array([300.0, 300.0, 0.0])
]

look_at = np.array([0.0, 0.0, 0.0])

# Iterate over each viewpoint position
for position in positions:
    # Create and configure each viewpoint
    viewpoint = Viewpoint.from_lookat(position, look_at)
    viewpoint.setDownsampleFactor(8.0)
    
    # Track the viewpoint in the VisibilityManager
    visibility_manager.trackViewpoint(viewpoint)

    # Perform raycasting on the viewpoint
    viewpoint.performRaycastingOnGPU(model)

    # Add viewpoint to the visualizer
    visualizer.addViewpoint(viewpoint, True, True)

    # Print novel coverage score
    novel_coverage_score = visibility_manager.computeNovelCoverageScore(viewpoint)
    print(f"Novel coverage score: {novel_coverage_score}")

# Retrieve visible voxels and coverage score from VisibilityManager
visible_voxels = visibility_manager.getVisibleVoxels()
coverage_score = visibility_manager.getCoverageScore()

# Output the number of visible voxels and coverage score
print(f"Number of visible voxels: {len(visible_voxels)}")
print(f"Coverage score: {coverage_score}")

# Set up voxel map visualization properties and render
base_color = np.array([1.0, 1.0, 1.0])  # White color
property_color = np.array([0.0, 1.0, 0.0])  # Green color
visualizer.addVoxelMapProperty(model, "visibility", base_color, property_color)

# Render the visualizer
visualizer.render()
