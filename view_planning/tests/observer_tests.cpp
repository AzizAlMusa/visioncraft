#include "visioncraft/model.h"
#include "visioncraft/viewpoint.h"
#include "visioncraft/visibility_manager.h"
#include "visioncraft/visualizer.h"
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <thread>
#include <chrono>


std::vector<std::shared_ptr<visioncraft::Viewpoint>> generateRandomViewpoints(int num_viewpoints, float sphere_radius) {
    std::vector<std::shared_ptr<visioncraft::Viewpoint>> viewpoints;

    // Generate viewpoints randomly on a sphere
    for (int i = 0; i < num_viewpoints; ++i) {
        // Generate random spherical coordinates
        float theta = static_cast<float>(rand()) / RAND_MAX * 2 * M_PI;  // Random azimuthal angle [0, 2π]
        float phi = static_cast<float>(rand()) / RAND_MAX * M_PI - M_PI / 2;  // Random polar angle [-π/2, π/2]

        // Convert spherical to Cartesian coordinates
        float x = sphere_radius * cos(phi) * cos(theta);
        float y = sphere_radius * cos(phi) * sin(theta);
        float z = sphere_radius * sin(phi);

        // Create the viewpoint and set its position
        Eigen::Vector3d position(x, y, z);
        Eigen::Vector3d look_at(0.0, 0.0, 0.0);  // Assume the viewpoints look towards the origin

        viewpoints.emplace_back(std::make_shared<visioncraft::Viewpoint>(position, look_at));
    }

    return viewpoints;
}


// Function to compute the attractive force
Eigen::Vector3d computeAttractiveForce(
    const visioncraft::Model& model, 
    const std::shared_ptr<visioncraft::Viewpoint>& viewpoint,
    float sigma, 
    int V_max) 
{
    Eigen::Vector3d F_attr = Eigen::Vector3d::Zero();
    const auto& voxelMap = model.getVoxelMap().getMap();

    for (const auto& kv : voxelMap) {
        const auto& voxel = kv.second;
        const auto& key = kv.first;

        int visibility = boost::get<int>(model.getVoxelProperty(key, "visibility"));
        float V_norm = static_cast<float>(visibility) / V_max;

        Eigen::Vector3d r = viewpoint->getPosition() - voxel.getPosition();
        float distance_squared = r.squaredNorm();
        float W = std::exp(-(distance_squared - 400.0 * 400.0)/ (2 * sigma * sigma));
        Eigen::Vector3d grad_W = -r * W / (sigma * sigma);

        F_attr += (1.0f - V_norm) * grad_W;
    }
    return F_attr;
}

// Function to compute the repulsive force
Eigen::Vector3d computeRepulsiveForce(
    const std::vector<std::shared_ptr<visioncraft::Viewpoint>>& viewpoints, 
    const std::shared_ptr<visioncraft::Viewpoint>& viewpoint, 
    float k_repel, 
    float alpha) 
{
    Eigen::Vector3d F_repel = Eigen::Vector3d::Zero();

    for (const auto& other_viewpoint : viewpoints) {
        if (viewpoint != other_viewpoint) {
            Eigen::Vector3d r = viewpoint->getPosition() - other_viewpoint->getPosition();
            float distance = r.norm() + 1e-5f; // Add small epsilon to prevent division by zero
            Eigen::Vector3d force = k_repel * r / std::pow(distance, alpha + 1.0f);
            F_repel += force;
        }
    }
    return F_repel;
}

// Function to update the viewpoint state
void updateViewpointState(
    const std::shared_ptr<visioncraft::Viewpoint>& viewpoint,
    const Eigen::Vector3d& new_position,
    float sphere_radius) 
{
    // Normalize to lie on sphere
    Eigen::Vector3d normalized_position = sphere_radius * new_position.normalized();

    // Update viewpoint
    viewpoint->setPosition(normalized_position);
    viewpoint->setLookAt(Eigen::Vector3d(0.0, 0.0, 0.0), -Eigen::Vector3d::UnitZ());

}

int main() {
    srand(time(nullptr)); // Initialize random seed

    // Initialize the visualizer
    visioncraft::Visualizer visualizer;
    visualizer.setBackgroundColor(Eigen::Vector3d(0.0, 0.0, 0.0));

    // Load the model
    visioncraft::Model model;
    std::cout << "Loading model..." << std::endl;
    model.loadModel("../models/gorilla.ply", 100000);
    std::cout << "Model loaded successfully." << std::endl;

    // Initialize visibility manager
    auto visibilityManager = std::make_shared<visioncraft::VisibilityManager>(model);
    std::cout << "VisibilityManager initialized." << std::endl;

    // Generate random viewpoints on the sphere
    float sphere_radius = 400.0f;  // Radius of the sphere
    int num_viewpoints = 8;       // Number of viewpoints to generate
    std::vector<std::shared_ptr<visioncraft::Viewpoint>> viewpoints = generateRandomViewpoints(num_viewpoints, sphere_radius);

    // Configure each viewpoint
    for (auto& viewpoint : viewpoints) {
        viewpoint->setDownsampleFactor(8.0);
  
        visibilityManager->trackViewpoint(viewpoint);
        
    }

    // Parameters
    float sigma = 100.0f;
    float k_repel = 5000.0f;
    float delta_t = 20.0f;
    float alpha = 1.0f;
    int V_max = num_viewpoints;
    int max_iterations = 500;

    // Main simulation loop
    for (int iter = 0; iter < max_iterations; ++iter) {
        std::cout << "Iteration " << iter << std::endl;

        // Step 2: Perform raycasting to update visibility
        auto start_raycasting = std::chrono::high_resolution_clock::now();
        for (auto& viewpoint : viewpoints) {

            viewpoint->performRaycastingOnGPU(model);
        }
        auto end_raycasting = std::chrono::high_resolution_clock::now();
        std::cout << "Raycasting time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_raycasting - start_raycasting).count()
                  << " ms" << std::endl;

        // Step 3: Compute forces and update viewpoints
        auto start_forces = std::chrono::high_resolution_clock::now();
        for (auto& viewpoint : viewpoints) {
            auto start_attr = std::chrono::high_resolution_clock::now();
            Eigen::Vector3d F_attr = computeAttractiveForce(model, viewpoint, sigma, V_max);
            auto end_attr = std::chrono::high_resolution_clock::now();

            auto start_repel = std::chrono::high_resolution_clock::now();
            Eigen::Vector3d F_repel = computeRepulsiveForce(viewpoints, viewpoint, k_repel, alpha);
            auto end_repel = std::chrono::high_resolution_clock::now();

             // Print the attractive and repulsive forces
            std::cout << "Attractive Force (F_attr): " << F_attr.transpose() << std::endl;
            std::cout << "Repulsive Force (F_repel): " << F_repel.transpose() << std::endl;

            Eigen::Vector3d F_total = F_attr + F_repel;
            Eigen::Vector3d n = viewpoint->getPosition().normalized();
            Eigen::Vector3d F_tangent = F_total - F_total.dot(n) * n;

            auto start_update = std::chrono::high_resolution_clock::now();
            Eigen::Vector3d new_position = viewpoint->getPosition() + delta_t * F_tangent;
            updateViewpointState(viewpoint, new_position, sphere_radius);
            auto end_update = std::chrono::high_resolution_clock::now();

            visualizer.addViewpoint(*viewpoint, false, true);
            // visualizer.updateViewpoint(*viewpoint, true, true);
            
            std::cout << "Attractive force time: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end_attr - start_attr).count()
                      << " ms, Repulsive force time: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end_repel - start_repel).count()
                      << " ms, Update time: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end_update - start_update).count()
                      << " ms" << std::endl;
        }
        auto end_forces = std::chrono::high_resolution_clock::now();
        std::cout << "Forces computation time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_forces - start_forces).count()
                  << " ms" << std::endl;

        // Step 4: Visualize voxel visibility
        visualizer.addVoxelMapProperty(model, "visibility");

        // Render the visualizer
        visualizer.render();

        // Remove voxel property visualization for next iteration
        visualizer.removeViewpoints();
        visualizer.removeVoxelMapProperty();

        // Add a small delay
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    return 0;
}
