from scipy.spatial.transform import Rotation as R
import numpy as np
import sys
import time
import random
import os
import trimesh

# Import visioncraft bindings
sys.path.append("../build/python_bindings")
from visioncraft_py import Model, Viewpoint, VisibilityManager, Visualizer

def generate_random_positions(radius, count):
    positions = []
    for _ in range(count):
        theta = random.uniform(0, 2 * np.pi)
        phi = random.uniform(0, np.pi)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        positions.append(np.array([x, y, z]))
    return positions

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return r, theta, phi

def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.array([x, y, z])

def perturb_position(position, radius, temperature):
    r, theta, phi = cartesian_to_spherical(*position)
    print(f"Temperature: {temperature:.3f}")    
    delta_theta = temperature * random.uniform(-0.005, 0.005)
    delta_phi = temperature * random.uniform(-0.005, 0.005)
    new_theta = theta + delta_theta
    new_phi = phi + delta_phi
    return spherical_to_cartesian(r, new_theta, new_phi)

def perturb_orientation(viewpoint, temperature):
    yaw, pitch, roll = viewpoint.getOrientationEuler()
    delta_yaw = temperature * random.uniform(-0.05, 0.05)
    delta_pitch = temperature * random.uniform(-0.05, 0.05)
    delta_roll = temperature * random.uniform(-0.05, 0.05)
    new_yaw, new_pitch, new_roll = yaw + delta_yaw, pitch + delta_pitch, roll + delta_roll
    orientation_matrix = R.from_euler('zyx', [new_yaw, new_pitch, new_roll], degrees=False).as_matrix()
    viewpoint.setOrientation(orientation_matrix)

def apply_perturbations(viewpoint, temperature, radius, perturb_position_flag, perturb_orientation_flag):
    if perturb_position_flag:
        original_position = viewpoint.getPosition()
        perturbed_position = perturb_position(original_position, radius, temperature)
        viewpoint.setPosition(perturbed_position)
    if perturb_orientation_flag:
        perturb_orientation(viewpoint, temperature)

# Initialize Visualizer, Model, and VisibilityManager
visualizer = Visualizer()
visualizer.initializeWindow("3D View")
visualizer.setBackgroundColor([0.0, 0.0, 0.0])


# Define the base directory for ShapeNet models
shapenet_dir = '/home/abdulaziz/playground/ShapeNet/ShapeNetCore/02691156/02691156'

# Specify the model ID you want to load
model_id = '3fd97395c1d1479b35cfde72d9f6a4cf'
model_path = os.path.join(shapenet_dir, model_id, 'models', 'model_normalized.obj')

# Define the output path for the converted .ply file
ply_path = os.path.join(shapenet_dir, model_id, 'models', 'model_normalized.ply')



# Now, load the .ply file with loadModel
model = Model()
print(f"Loading model from {ply_path}")
model.loadModel("../models/gorilla.ply", 50000)
print(f"Model loaded successfully")
visibility_manager = VisibilityManager(model)

# Set up for greedy algorithm
target_coverage, achieved_coverage = 0.8, visibility_manager.getCoverageScore()
selected_viewpoints = []
start_time = time.time()

# Greedy algorithm for initial coverage
import pdb; pdb.set_trace()
while achieved_coverage < target_coverage:
    best_viewpoint, best_coverage_increment = None, 0.0
    for position in generate_random_positions(3, 10):
        viewpoint = Viewpoint.from_lookat(position, np.array([0.0, 0.0, 0.0]))
        viewpoint.setDownsampleFactor(8.0)
        viewpoint.setNearPlane(0.1)
        viewpoint.setFarPlane(5.0)
        
        visibility_manager.trackViewpoint(viewpoint)
        viewpoint.performRaycastingOnGPU(model)
        
        coverage_increment = visibility_manager.computeNovelCoverageScore(viewpoint)
        visibility_manager.untrackViewpoint(viewpoint)

        if coverage_increment > best_coverage_increment:
            best_coverage_increment, best_viewpoint = coverage_increment, viewpoint

    if best_viewpoint:
        visibility_manager.trackViewpoint(best_viewpoint)
        best_viewpoint.performRaycastingOnGPU(model)
        
        achieved_coverage += best_coverage_increment
        selected_viewpoints.append(best_viewpoint)

print(f"Initial Coverage Score: {achieved_coverage:.3f} (Target: 0.8)")
print(f"Total Viewpoints Selected: {len(selected_viewpoints)}")

# Render the result of the greedy algorithm phase for verification
visualizer.addVoxelMapProperty(model, "visibility", [1.0, 1.0, 1.0], [0.0, 1.0, 0.0])
for viewpoint in selected_viewpoints:
    visualizer.addViewpoint(viewpoint, True, True)

visualizer.render()

# Simulated Annealing
temperature, cooling_rate = 1.0, 0.99  # Slower cooling
initial_total_coverage = visibility_manager.getCoverageScore()
perturb_position_flag, perturb_orientation_flag = False, True

while temperature > 0.01:
    for viewpoint in selected_viewpoints:
        original_position = viewpoint.getPosition() if perturb_position_flag else None
        original_orientation = viewpoint.getOrientationMatrix() if perturb_orientation_flag else None

        apply_perturbations(viewpoint, temperature, 400, perturb_position_flag, perturb_orientation_flag)
        viewpoint.performRaycastingOnGPU(model)
        total_coverage_after_perturb = visibility_manager.getCoverageScore()

        if total_coverage_after_perturb > initial_total_coverage or \
           random.uniform(0, 1) < np.exp((total_coverage_after_perturb - initial_total_coverage) / (0.5 * temperature)):
            print(f"Accepted Perturbation - Coverage: {total_coverage_after_perturb:.3f}")
            initial_total_coverage = total_coverage_after_perturb
        else:
            if perturb_position_flag and original_position is not None:
                viewpoint.setPosition(original_position)
            if perturb_orientation_flag and original_orientation is not None:
                viewpoint.setOrientation(original_orientation)
            viewpoint.performRaycastingOnGPU(model)
            print("Reverted Perturbation")

    temperature *= cooling_rate

# Final Score
final_coverage_score = visibility_manager.getCoverageScore()
print(f"Final Coverage Score after Annealing: {final_coverage_score:.3f}")
print(f"Time taken: {time.time() - start_time:.2f} seconds")
