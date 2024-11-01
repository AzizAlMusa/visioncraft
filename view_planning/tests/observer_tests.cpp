#include "visioncraft/model.h"
#include "visioncraft/viewpoint.h"
#include "visioncraft/visibility_manager.h"
#include "visioncraft/visualizer.h"
#include <memory>
#include <iostream>
#include <vector>

int main() {
    visioncraft::Visualizer visualizer;
    visualizer.setBackgroundColor(Eigen::Vector3d(0.0, 0.0, 0.0));

    // Create a model instance and load a model file
    visioncraft::Model model;
    model.loadModel("../models/cube.ply", 50000);

    // Create a VisibilityManager with the model
    auto visibilityManager = std::make_shared<visioncraft::VisibilityManager>(model);

    // Define positions for six viewpoints
    std::vector<Eigen::Vector3d> positions = {
        Eigen::Vector3d(400.0, 0.0, 0.0),
        Eigen::Vector3d(-400.0, 0.0, 0.0),
        Eigen::Vector3d(0.0, 400.0, 0.0),
        Eigen::Vector3d(0.0, -400.0, 0.0),
        Eigen::Vector3d(0.0, 0.0, 400.0),
        Eigen::Vector3d(0.0, 0.0, -400.0),
        Eigen::Vector3d(300.0, 300.0, 0.0)
    };

    Eigen::Vector3d lookAt(0.0, 0.0, 0.0);

    // Iterate over each viewpoint position
    for (const auto& position : positions) {
        auto viewpoint = std::make_shared<visioncraft::Viewpoint>(position, lookAt);
        viewpoint->setDownsampleFactor(8.0);

        // Track the viewpoint in the VisibilityManager
        visibilityManager->trackViewpoint(viewpoint);

        // Perform raycasting on the viewpoint
        viewpoint->performRaycastingOnGPU(model);
 
        visualizer.addViewpoint(*viewpoint, true, true);
    }

    // Get visible voxels tracked by the VisibilityManager
    auto visibleVoxels = visibilityManager->getVisibleVoxels();

    auto coverageScore = visibilityManager->getCoverageScore();

    // Output the number of visible voxels
    std::cout << "Number of visible voxels: " << visibleVoxels.size() << std::endl;
    std::cout << "Coverage score: " << coverageScore << std::endl;

    Eigen::Vector3d baseColor(1.0, 1.0, 1.0);  // Green color
    Eigen::Vector3d propertyColor(0.0, 1.0, 0.0);  // Green color
    visualizer.addVoxelMapProperty(model, "visibility",baseColor, propertyColor);
    visualizer.render();
    return 0;
}
