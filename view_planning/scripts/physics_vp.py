import numpy as np
import time
import sys
import os

sys.path.append(os.path.abspath("../build/python_bindings"))

from visioncraft_py import Model, Viewpoint, Visualizer, VisibilityManager

def generate_random_viewpoints(num_viewpoints, sphere_radius):
    viewpoints = []

    for _ in range(num_viewpoints):
        theta = np.random.uniform(0, 2 * np.pi)  # Random azimuthal angle [0, 2π]
        phi = np.random.uniform(-np.pi / 2, np.pi / 2)  # Random polar angle [-π/2, π/2]

        x = sphere_radius * np.cos(phi) * np.cos(theta)
        y = sphere_radius * np.cos(phi) * np.sin(theta)
        z = sphere_radius * np.sin(phi)

        position = np.array([x, y, z])
        look_at = np.array([0.0, 0.0, 0.0])

        viewpoints.append(Viewpoint.from_lookat(position, look_at))
    
    return viewpoints

def compute_attractive_force(model, viewpoint, sigma, V_max):
    F_attr = np.zeros(3)
    voxel_map = model.getVoxelMap()
    for key, voxel in voxel_map.items():
        
        visibility = model.getMetaVoxelProperty(key, 'visibility')
        V_norm = visibility / V_max

        r = viewpoint.getPosition() - voxel.getPosition()
        distance_squared = np.dot(r, r)
        W = np.exp(-(distance_squared - 400.0 ** 2) / (2 * sigma ** 2))
        grad_W = -r * W / (sigma ** 2)

        F_attr += (1.0 - V_norm) * grad_W

    return F_attr

def compute_repulsive_force(viewpoints, viewpoint, k_repel, alpha):
    F_repel = np.zeros(3)
    
    for other_viewpoint in viewpoints:
        if viewpoint is not other_viewpoint:
            r = viewpoint.getPosition() - other_viewpoint.getPosition()
            distance = np.linalg.norm(r) + 1e-5
            force = k_repel * r / (distance ** (alpha + 1.0))
            F_repel += force

    return F_repel

def update_viewpoint_state(viewpoint, new_position, sphere_radius):
    normalized_position = sphere_radius * (new_position / np.linalg.norm(new_position))
    viewpoint.setPosition(normalized_position)
    viewpoint.setLookAt(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]))

def main():
    np.random.seed(int(time.time()))

    visualizer = Visualizer()
    visualizer.setBackgroundColor(np.array([0.0, 0.0, 0.0]))

    model = Model()
    print("Loading model...")
    model.loadModel("../models/gorilla.ply", 100000)
    print("Model loaded successfully.")

    visibility_manager = VisibilityManager(model)
    print("VisibilityManager initialized.")

    sphere_radius = 400.0
    num_viewpoints = 8
    viewpoints = generate_random_viewpoints(num_viewpoints, sphere_radius)

    for viewpoint in viewpoints:
        viewpoint.setDownsampleFactor(8.0)
        visibility_manager.trackViewpoint(viewpoint)

    sigma = 100.0
    k_repel = 5000.0
    delta_t = 20.0
    alpha = 1.0
    V_max = num_viewpoints
    max_iterations = 500

    for iteration in range(max_iterations):
        print(f"Iteration {iteration}")

        start_raycasting = time.time()
        for viewpoint in viewpoints:
            viewpoint.performRaycastingOnGPU(model)
        end_raycasting = time.time()
        print(f"Raycasting time: {end_raycasting - start_raycasting:.2f} s")

        start_forces = time.time()
        for viewpoint in viewpoints:
            F_attr = compute_attractive_force(model, viewpoint, sigma, V_max)
            F_repel = compute_repulsive_force(viewpoints, viewpoint, k_repel, alpha)

            print(f"Attractive Force (F_attr): {F_attr}")
            print(f"Repulsive Force (F_repel): {F_repel}")

            F_total = F_attr + F_repel
            n = viewpoint.getPosition() / np.linalg.norm(viewpoint.getPosition())
            F_tangent = F_total - np.dot(F_total, n) * n

            new_position = viewpoint.getPosition() + delta_t * F_tangent
            update_viewpoint_state(viewpoint, new_position, sphere_radius)

            visualizer.addViewpoint(viewpoint, showFrustum=False, showAxes=True)
        end_forces = time.time()
        print(f"Forces computation time: {end_forces - start_forces:.2f} s")

        visualizer.addVoxelMapProperty(model, "visibility")
        visualizer.render()
        visualizer.removeViewpoints()
        visualizer.removeVoxelMapProperty()

        time.sleep(0.001)

if __name__ == "__main__":
    main()
