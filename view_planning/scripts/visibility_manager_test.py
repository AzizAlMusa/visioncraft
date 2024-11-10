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
test_path = "/home/abdulaziz/playground/ShapeNet/ShapeNetCore/02691156/02691156/3fd97395c1d1479b35cfde72d9f6a4cf/models/model_normalized.ply"
main_path = "../models/gorilla.ply"
model.loadModel(main_path, 20000)
visibility_manager = VisibilityManager(model)

# Greedy algorithm setup
target_coverage, achieved_coverage = 0.998, visibility_manager.getCoverageScore()
selected_viewpoints = []
start_time = time.time()

# Greedy algorithm loop
while achieved_coverage < target_coverage:
    print("Starting a new iteration of the greedy algorithm loop.")
    best_viewpoint, best_coverage_increment = None, 0.0

    # Test each viewpoint in the batch
    print("Generating random positions and evaluating viewpoints...")
    random_positions = generate_random_positions(400, 10)
    for index, position in enumerate(random_positions):
        print(f"Evaluating viewpoint {index + 1}/{len(random_positions)} at position: {position}")
        
        viewpoint = Viewpoint.from_lookat(position, [0.0, 0.0, 0.0])
        viewpoint.setDownsampleFactor(8.0)
        # viewpoint.setNearPlane(0.1)
        # viewpoint.setFarPlane(5.0)

        visibility_manager.trackViewpoint(viewpoint)
        print("Performing raycasting on GPU...")
        viewpoint.performRaycastingOnGPU(model)
        
        coverage_increment = visibility_manager.computeNovelCoverageScore(viewpoint)
        print(f"Coverage increment for this viewpoint: {coverage_increment:.4f}")
        
        visibility_manager.untrackViewpoint(viewpoint)

        if coverage_increment > best_coverage_increment:
            best_coverage_increment, best_viewpoint = coverage_increment, viewpoint
            print(f"New best viewpoint found with increment: {best_coverage_increment:.4f}")

    # Track the best viewpoint and update coverage
    if best_viewpoint:
        print(f"Tracking the best viewpoint with coverage increment: {best_coverage_increment:.4f}")
        visibility_manager.trackViewpoint(best_viewpoint)
        print("Performing raycasting on GPU for the best viewpoint...")
        best_viewpoint.performRaycastingOnGPU(model)
        
        achieved_coverage += best_coverage_increment
        selected_viewpoints.append(best_viewpoint)
        print(f"Coverage updated: {achieved_coverage:.4f} / {target_coverage:.4f}")
    else:
        print("No suitable viewpoint found in this iteration.")
    
    print("End of current iteration.\n")

    
    

# Print results
print(f"\nFinal Coverage Score: {visibility_manager.getCoverageScore()}")
print(f"Total Viewpoints Selected: {len(selected_viewpoints)}")
print(f"Time taken for greedy algorithm (excluding rendering): {time.time() - start_time:.2f} seconds")

# Render selected viewpoints
visualizer.addVoxelMapProperty(model, "visibility", [1.0, 1.0, 1.0], [0.0, 1.0, 0.0])
for viewpoint in selected_viewpoints:
    visualizer.addViewpoint(viewpoint, True, True)

visualizer.render()
