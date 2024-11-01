import numpy as np
import sys
import os
import random

# Adjust path to locate bindings if necessary
sys.path.append(os.path.abspath("../build/python_bindings"))
from visioncraft_py import Model, Viewpoint, VisibilityManager, Visualizer

# Function to generate random positions on the surface of a sphere
def generate_random_positions(radius, count):
    positions = []
    for _ in range(count):
        # Random spherical coordinates
        theta = random.uniform(0, 2 * np.pi)  # azimuthal angle
        phi = random.uniform(0, np.pi)        # polar angle
        
        # Convert spherical coordinates to Cartesian coordinates
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        
        positions.append(np.array([x, y, z]))
    return positions

# Initialize the Visualizer
visualizer = Visualizer()
visualizer.initializeWindow("3D View")
visualizer.setBackgroundColor(np.array([0.0, 0.0, 0.0]))  # Set background color to black

# Load the model
model = Model()
model.loadModel("../models/cube.ply", 50000)

# Create the VisibilityManager
visibility_manager = VisibilityManager(model)

# Generate random positions for viewpoints within a radius of 80
positions = generate_random_positions(80, 7)  # Generate 7 random positions
lookAt = np.array([0.0, 0.0, 0.0])

# Iterate over each viewpoint position
for position in positions:
    # Create a viewpoint with the specified position and lookAt point
    viewpoint = Viewpoint.from_lookat(position, lookAt)
    viewpoint.setDownsampleFactor(8.0)  # Set downsample factor
    viewpoint.setNearPlane(10)         # Set near plane distance
    viewpoint.setFarPlane(100)         # Set far plane distance
    # Track the viewpoint in the VisibilityManager
    visibility_manager.trackViewpoint(viewpoint)

    # Perform raycasting on the viewpoint
    viewpoint.performRaycastingOnGPU(model)

    # Visualize the viewpoint with frustum and rays
    visualizer.addViewpoint(viewpoint, True, True)

# Get the visible voxels tracked by the VisibilityManager
visible_voxels = visibility_manager.getVisibleVoxels()
coverage_score = visibility_manager.getCoverageScore()

# Output the number of visible voxels and coverage score
print(f"Number of visible voxels: {len(visible_voxels)}")
print(f"Coverage score: {coverage_score}")

# Visualize the voxel map with a 'visibility' property (assuming visibility property is tracked)
base_color = np.array([1.0, 1.0, 1.0])      # White color
property_color = np.array([0.0, 1.0, 0.0])   # Green color
visualizer.addVoxelMapProperty(model, "visibility", base_color, property_color)

# Start the rendering loop
visualizer.render()
