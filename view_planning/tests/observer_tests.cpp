#include "visioncraft/model.h"
#include "visioncraft/viewpoint.h"
#include "visioncraft/visibility_manager.h"
#include "visioncraft/visualizer.h"
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>

// Function to calculate the attraction potential field for a voxel
float computeField(const Eigen::Vector3d& voxelPosition, const Eigen::Vector3d& viewpointPosition, int visibility) {
    // Weighting function w(p): Euclidean distance between viewpoint and voxel
    float weight = static_cast<float>((voxelPosition - viewpointPosition).norm());

    // Calculate attraction potential as w(p) * phi(v_i, p), where phi is visibility (1 if visible, 0 otherwise)
    return weight * (1 - visibility);
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
    // Generate a single viewpoint
    Eigen::Vector3d viewpointPosition(400.0, 0.0, 0.0); // Example viewpoint on the bounding sphere
    auto viewpoint = std::make_shared<visioncraft::Viewpoint>(viewpointPosition, Eigen::Vector3d(0.0, 0.0, 0.0));
    viewpoint->setDownsampleFactor(8.0);
    
    visibilityManager->trackViewpoint(viewpoint);

    // Perform raycasting for the viewpoint
    std::cout << "Performing raycasting for the viewpoint..." << std::endl;
    viewpoint->performRaycastingOnGPU(model);

    // Access the voxel map and iterate through its entries
    const auto& voxelMap = model.getVoxelMap().getMap();

    model.addVoxelProperty("attractive_field", 0.0);  // Initialize the attractive field property for all voxels
    for (auto it = voxelMap.begin(); it != voxelMap.end(); ++it) {
        const auto& key = it->first;
        auto& voxel = it->second; // Non-const reference to allow modification

        // Retrieve visibility status (1 if visible, 0 if not)
        int visibility = boost::get<int>(model.getVoxelProperty(key, "visibility"));

        // Calculate the attraction potential field for the current voxel using Euclidean distance as the weight
        float attractiveField = computeField(voxel.getPosition(), viewpointPosition, visibility);

        // Set the attractive field as a property in the MetaVoxel
        model.setVoxelProperty(key, "attractive_field", attractiveField);
    }

    // Visualize the updated model with attractive fields
    visualizer.addVoxelMapProperty(model, "attractive_field", Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d(1.0, 0.0, 0.0));
    visualizer.addViewpoint(*viewpoint, true, true);
    visualizer.render();

    return 0;
}
