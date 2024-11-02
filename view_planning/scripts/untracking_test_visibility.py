import numpy as np
import sys
import os
import time

# Adjust path to locate bindings if necessary
sys.path.append(os.path.abspath("../build/python_bindings"))
from visioncraft_py import Model, Viewpoint, VisibilityManager, Visualizer

# Function to create and track three viewpoints
def create_and_track_viewpoints(model, visibility_manager):
    # Define positions for the viewpoints
    position_x = np.array([400.0, 0.0, 0.0])  # Viewpoint on the x-axis
    position_y = np.array([0.0, 400.0, 0.0])  # Viewpoint on the y-axis
    position_z = np.array([0.0, 0.0, 400.0])  # Viewpoint on the z-axis
    look_at = np.array([0.0, 0.0, 0.0])      # Look at the origin

    # Create viewpoints
    viewpoint_x = Viewpoint.from_lookat(position_x, look_at)
    viewpoint_y = Viewpoint.from_lookat(position_y, look_at)
    viewpoint_z = Viewpoint.from_lookat(position_z, look_at)

    # Set parameters for all viewpoints
    for viewpoint in [viewpoint_x, viewpoint_y, viewpoint_z]:
        viewpoint.setDownsampleFactor(8.0)
        # viewpoint.setNearPlane(10)
        # viewpoint.setFarPlane(100)
        visibility_manager.trackViewpoint(viewpoint)  # Track each viewpoint

    # Perform raycasting for all viewpoints
    for viewpoint in [viewpoint_x, viewpoint_y, viewpoint_z]:
        viewpoint.performRaycastingOnGPU(model)

    return viewpoint_x, viewpoint_y, viewpoint_z

# Function to render and visualize coverage
def render_and_visualize(visualizer, visibility_manager):
    # Get visible voxels and coverage score
    visible_voxels = visibility_manager.getVisibleVoxels()
    coverage_score = visibility_manager.getCoverageScore()

    # Output the number of visible voxels and coverage score
    print(f"Number of visible voxels: {len(visible_voxels)}")
    print(f"Coverage score: {coverage_score}")

    # Visualize the voxel map with a 'visibility' property
    base_color = np.array([1.0, 1.0, 1.0])      # White color
    property_color = np.array([0.0, 1.0, 0.0])   # Green color
    visualizer.addVoxelMapProperty(model, "visibility", base_color, property_color)

    # Start rendering loop
    visualizer.render()

# Initialize the Visualizer
visualizer = Visualizer()
visualizer.initializeWindow("3D View")
visualizer.setBackgroundColor(np.array([0.0, 0.0, 0.0]))  # Set background color to black

# Load the model
model = Model()
model.loadModel("../models/cube.ply", 50000)

# Create the VisibilityManager
visibility_manager = VisibilityManager(model)

# Create and track viewpoints
viewpoint_x, viewpoint_y, viewpoint_z = create_and_track_viewpoints(model, visibility_manager)

# Untrack one of the viewpoints (e.g., the one on the x-axis)
coverage_score = visibility_manager.getCoverageScore()
print(f"Coverage score: {coverage_score}")
visibility_manager.untrackViewpoint(viewpoint_x)

# Render and visualize coverage
render_and_visualize(visualizer, visibility_manager)
