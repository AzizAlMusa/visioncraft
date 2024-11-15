import numpy as np
import sys
import os
import time

# Adjust path to locate bindings if necessary
sys.path.append(os.path.abspath("../build/python_bindings"))
from visioncraft_py import Model, Visualizer


# Initialize the Visualizer
visualizer = Visualizer()
visualizer.initializeWindow("3D View")
visualizer.setBackgroundColor(np.array([0.0, 0.0, 0.0]))

# Load model
model = Model()
model.loadModel("../models/gorilla.ply", 50000)
model.addVoxelProperty("radius", 0.0)  # Initialize radius property for all voxels

for key, voxel in model.getVoxelMap().items():

    voxel_position = voxel.getPosition()  # get the voxel position
    radius = np.linalg.norm(voxel_position)  # Compute the Euclidean distance from the origin
    model.setVoxelProperty(key, "radius", radius)  # Set the radius property for visualization


# Visualize the 'radius' property
baseColor = np.array([0.0, 0.0, 1.0])      # Base color for the voxel map
propertyColor = np.array([1.0, 0.0, 0.0])   # Color to represent increasing distance from origin
#time this function
visualizer.addVoxelMapProperty(model, "radius", baseColor, propertyColor)


# Start rendering loop
visualizer.render()
