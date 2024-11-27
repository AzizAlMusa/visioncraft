#include "visioncraft/model.h"
#include "visioncraft/viewpoint.h"
#include "visioncraft/visibility_manager.h"
#include "visioncraft/visualizer.h"
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <thread>
#include <chrono>
#include <memory>

// Update viewpoint state
void updateViewpointState(
    const std::shared_ptr<visioncraft::Viewpoint>& viewpoint,
    const Eigen::Vector3d& new_position,
    float sphere_radius) 
{
    Eigen::Vector3d normalized_position = sphere_radius * new_position.normalized();
    viewpoint->setPosition(normalized_position);
    viewpoint->setLookAt(Eigen::Vector3d(0.0, 0.0, 0.0), -Eigen::Vector3d::UnitZ());
}

int main() {
    // Initialize visualizer
    visioncraft::Visualizer visualizer;
    visualizer.setBackgroundColor(Eigen::Vector3d(0.0, 0.0, 0.0));

    // Load model
    visioncraft::Model model;
    model.loadModel("../models/cat.ply", 100000);

    // Create VisibilityManager
    auto visibilityManager = std::make_shared<visioncraft::VisibilityManager>(model);

    // Prepare to load viewpoints from CSV
    std::ifstream viewpoint_csv("viewpoint_positions.csv");
    if (!viewpoint_csv.is_open()) {
        std::cerr << "Error: Unable to open viewpoint_positions.csv" << std::endl;
        return -1;
    }

    // Parse the CSV header
    std::string header_line;
    std::getline(viewpoint_csv, header_line);

    // Prepare viewpoint instances
    int num_viewpoints = 0;
    std::vector<std::shared_ptr<visioncraft::Viewpoint>> viewpoints;
    std::vector<std::vector<Eigen::Vector3d>> positions_per_timestep;

    // Load positions into a map
    std::string line;
    int current_timestep = -1;
    std::vector<Eigen::Vector3d> current_positions;

    while (std::getline(viewpoint_csv, line)) {
        std::istringstream iss(line);
        std::string token;
        int timestep, viewpoint_id;
        double x, y, z;

        std::getline(iss, token, ',');
        timestep = std::stoi(token);
        std::getline(iss, token, ',');
        viewpoint_id = std::stoi(token);
        std::getline(iss, token, ',');
        x = std::stod(token);
        std::getline(iss, token, ',');
        y = std::stod(token);
        std::getline(iss, token, ',');
        z = std::stod(token);

        if (current_timestep != timestep) {
            if (!current_positions.empty()) {
                positions_per_timestep.push_back(current_positions);
                current_positions.clear();
            }
            current_timestep = timestep;
        }
        current_positions.emplace_back(x, y, z);
    }
    if (!current_positions.empty()) {
        positions_per_timestep.push_back(current_positions);
    }

    viewpoint_csv.close();
    num_viewpoints = positions_per_timestep[0].size();

    // Initialize viewpoints at their first position
    Eigen::Vector3d lookAt(0.0, 0.0, 0.0);
    for (int i = 0; i < num_viewpoints; ++i) {
        auto viewpoint = std::make_shared<visioncraft::Viewpoint>(positions_per_timestep[0][i], lookAt);
        viewpoint->setDownsampleFactor(8.0);
        visibilityManager->trackViewpoint(viewpoint);
        viewpoints.push_back(viewpoint);
    }

    // Animation loop
    for (const auto& positions : positions_per_timestep) {
        for (size_t i = 0; i < viewpoints.size(); ++i) {
            // Update the viewpoint's position
            updateViewpointState(viewpoints[i], positions[i], 400.0f);

            // Perform raycasting for the updated viewpoint
            viewpoints[i]->performRaycastingOnGPU(model);

            // Add the updated viewpoint to the visualizer
            visualizer.addViewpoint(*viewpoints[i], false, true);
            visualizer.updateViewpoint(*viewpoints[i], true, true);
        }

        // Add voxel map for visibility
        visualizer.addVoxelMapProperty(model, "visibility");

        // Render the current state
        visualizer.render();

        // Sleep for animation delay
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        // Clean up for the next frame
        // visualizer.removeViewpoints();
        visualizer.removeVoxelMapProperty();
    }

    return 0;
}

