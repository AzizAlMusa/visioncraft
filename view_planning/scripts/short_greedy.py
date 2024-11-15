import numpy as np
import sys
import time
import random

# Import visioncraft bindings
sys.path.append("../build/python_bindings")
from visioncraft_py import Model, Viewpoint, VisibilityManager, Visualizer

def generate_random_positions(radius, count):
    return [
        np.array([radius * np.sin(phi) * np.cos(theta), 
                  radius * np.sin(phi) * np.sin(theta), 
                  radius * np.cos(phi)])
        for theta, phi in zip(np.random.uniform(0, 2 * np.pi, count), 
                              np.random.uniform(0, np.pi, count))
    ]

# Initialize model and visibility manager
visualizer = Visualizer()
visualizer.initializeWindow("3D View")
visualizer.setBackgroundColor([0.0, 0.0, 0.0])

model = Model()
model.loadModel("../models/gorilla.ply", 50000)
visibility_manager = VisibilityManager(model)

# Greedy algorithm setup
target_coverage, achieved_coverage = 0.995, visibility_manager.getCoverageScore()
selected_viewpoints = []
start_time = time.time()

# Greedy algorithm loop
while achieved_coverage < target_coverage:
    best_viewpoint, best_coverage_increment = None, 0.0

    # Test each viewpoint in the batch
    for position in generate_random_positions(400, 10):
        viewpoint = Viewpoint.from_lookat(position, [0.0, 0.0, 0.0])
        viewpoint.setNearPlane(300)
        viewpoint.setFarPlane(900.0)
        viewpoint.setDownsampleFactor(8.0)
        
        visibility_manager.trackViewpoint(viewpoint)
        viewpoint.performRaycastingOnGPU(model)
        
        coverage_increment = visibility_manager.computeNovelCoverageScore(viewpoint)
        visibility_manager.untrackViewpoint(viewpoint)

        if coverage_increment > best_coverage_increment:
            best_coverage_increment, best_viewpoint = coverage_increment, viewpoint

    # Track the best viewpoint and update coverage
    if best_viewpoint:
        visibility_manager.trackViewpoint(best_viewpoint)
        best_viewpoint.performRaycastingOnGPU(model)
        
        achieved_coverage += best_coverage_increment
        selected_viewpoints.append(best_viewpoint)

# Print results
print(f"\nFinal Coverage Score: {visibility_manager.getCoverageScore()}")
print(f"Total Viewpoints Selected: {len(selected_viewpoints)}")
print(f"Time taken for greedy algorithm (excluding rendering): {time.time() - start_time:.2f} seconds")

# Render selected viewpoints
visualizer.addVoxelMapProperty(model, "visibility", [1.0, 1.0, 1.0], [0.0, 1.0, 0.0])
for viewpoint in selected_viewpoints:
    visualizer.addViewpoint(viewpoint, True, True)

visualizer.render()