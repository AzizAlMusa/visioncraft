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
    const Eigen::Quaterniond& orientation,
    float sphere_radius) 
{
    Eigen::Vector3d normalized_position = sphere_radius * new_position.normalized();
    viewpoint->setPosition(normalized_position);
    viewpoint->setOrientation(orientation);
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
    std::ifstream viewpoint_csv("viewpoint_data.csv");
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
    std::vector<std::vector<std::tuple<Eigen::Vector3d, Eigen::Quaterniond>>> positions_per_timestep;

    // Load positions and quaternions into a map
    std::string line;
    int current_timestep = -1;
    std::vector<std::tuple<Eigen::Vector3d, Eigen::Quaterniond>> current_positions;

    while (std::getline(viewpoint_csv, line)) {
        std::istringstream iss(line);
        std::string token;
        int timestep, viewpoint_id;
        double x, y, z, qw, qx, qy, qz;

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
        std::getline(iss, token, ',');
        qw = std::stod(token);  // Quaternion w
        std::getline(iss, token, ',');
        qx = std::stod(token);  // Quaternion x
        std::getline(iss, token, ',');
        qy = std::stod(token);  // Quaternion y
        std::getline(iss, token, ',');
        qz = std::stod(token);  // Quaternion z

        Eigen::Quaterniond quaternion(qw, qx, qy, qz);

        if (current_timestep != timestep) {
            if (!current_positions.empty()) {
                positions_per_timestep.push_back(current_positions);
                current_positions.clear();
            }
            current_timestep = timestep;
        }
        current_positions.emplace_back(Eigen::Vector3d(x, y, z), quaternion);
    }
    if (!current_positions.empty()) {
        positions_per_timestep.push_back(current_positions);
    }

    viewpoint_csv.close();
    num_viewpoints = positions_per_timestep[0].size();

    // Initialize viewpoints at their first position
    Eigen::Vector3d lookAt(0.0, 0.0, 0.0);
    for (int i = 0; i < num_viewpoints; ++i) {
        auto viewpoint = std::make_shared<visioncraft::Viewpoint>(
            std::get<0>(positions_per_timestep[0][i]), lookAt);
        viewpoint->setOrientation(std::get<1>(positions_per_timestep[0][i]));
        viewpoint->setDownsampleFactor(8.0);
        visibilityManager->trackViewpoint(viewpoint);
        viewpoints.push_back(viewpoint);
    }

    // Animation loop
    for (const auto& positions : positions_per_timestep) {
        // Time measurement for each frame
        auto frame_start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < viewpoints.size(); ++i) {
            // Measure time for updating the viewpoint
            auto update_start = std::chrono::high_resolution_clock::now();
            updateViewpointState(viewpoints[i], std::get<0>(positions[i]), std::get<1>(positions[i]), 400.0f);
            auto update_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> update_duration = update_end - update_start;
            std::cout << "Update Viewpoint " << i << " took " << update_duration.count() << " seconds." << std::endl;

            // Measure time for raycasting
            auto raycast_start = std::chrono::high_resolution_clock::now();
            viewpoints[i]->performRaycastingOnGPU(model);
            auto raycast_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> raycast_duration = raycast_end - raycast_start;
            std::cout << "Raycasting for Viewpoint " << i << " took " << raycast_duration.count() << " seconds." << std::endl;

            // Measure time for adding viewpoint to visualizer
            auto add_viewpoint_start = std::chrono::high_resolution_clock::now();
            // visualizer.addViewpoint(*viewpoints[i], false, true);
            visualizer.updateViewpoint(*viewpoints[i], true, true);
            auto add_viewpoint_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> add_viewpoint_duration = add_viewpoint_end - add_viewpoint_start;
            std::cout << "Adding Viewpoint " << i << " to visualizer took " << add_viewpoint_duration.count() << " seconds." << std::endl;
        }

        // Add voxel map for visibility
        auto voxel_map_start = std::chrono::high_resolution_clock::now();
        visualizer.addVoxelMapProperty(model, "visibility");
        auto voxel_map_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> voxel_map_duration = voxel_map_end - voxel_map_start;
        std::cout << "Adding voxel map took " << voxel_map_duration.count() << " seconds." << std::endl;

        // Render the current state
        auto render_start = std::chrono::high_resolution_clock::now();
        visualizer.render();
        auto render_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> render_duration = render_end - render_start;
        std::cout << "Rendering took " << render_duration.count() << " seconds." << std::endl;

        // Sleep for animation delay
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

        // Clean up for the next frame
        visualizer.removeViewpoints();
        visualizer.removeVoxelMapProperty();

        // Measure total frame time
        auto frame_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> frame_duration = frame_end - frame_start;
        std::cout << "Frame processing took " << frame_duration.count() << " seconds." << std::endl;
    }

    return 0;
}
