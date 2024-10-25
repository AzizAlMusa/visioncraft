#include "visioncraft/model_loader.h"
#include "visioncraft/viewpoint.h"
#include "visioncraft/visualizer.h"
#include <octomap/ColorOcTree.h>
#include <chrono>
#include <iostream>
#include <thread>




int main() {




    // Create model loader and load a model
    auto model_loader = std::make_shared<visioncraft::ModelLoader>();

    if (!model_loader->loadModel("../models/cube.ply", 50000)) {
        std::cerr << "Failed to load model with default parameters." << std::endl;
        return -1;
    }

    // Create a viewpoint using lookAt
    Eigen::Vector3d position(100.0, 100, 100);
    Eigen::Vector3d lookAt(0.0, 0.0, 0.0);
    Eigen::Vector3d up(0.0, 0.0, -1);

    visioncraft::Viewpoint viewpoint(position, lookAt, up);

    viewpoint.setPosition(position);
    viewpoint.setLookAt(lookAt, up);

    viewpoint.setNearPlane(10.0);
    viewpoint.setFarPlane(150.0);

    auto octomap = model_loader->getOctomap();

    // Measure time for raycasting without multithreading
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<bool, Eigen::Vector3d>> hit_results = viewpoint.performRaycasting(octomap, false); // No multithreading
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "Raycasting without multithreading took: " << duration.count() << " seconds." << std::endl;

    // Measure time for raycasting with multithreading
    start_time = std::chrono::high_resolution_clock::now();
    hit_results = viewpoint.performRaycasting(octomap, true); // With multithreading
    end_time = std::chrono::high_resolution_clock::now();
    duration = end_time - start_time;
    std::cout << "Raycasting with multithreading took: " << duration.count() << " seconds." << std::endl;

    // Create visualizer and display results
    auto hit_lines = std::make_shared<open3d::geometry::LineSet>();

    for (const auto& hit : hit_results) {
        if (hit.first) {
            hit_lines->points_.push_back(position);
            hit_lines->points_.push_back(hit.second);

            hit_lines->lines_.emplace_back(hit_lines->points_.size() - 2, hit_lines->points_.size() - 1);
            hit_lines->colors_.push_back(Eigen::Vector3d(0.0, 1.0, 0.0));
        }
    }

    visioncraft::Visualizer visualizer;
    visualizer.addVoxelGrid(model_loader->getVoxelGrid());
    visualizer.addLineSet(hit_lines);
    visualizer.addViewpoint(viewpoint);

    visualizer.show(true);

    return 0;
}
