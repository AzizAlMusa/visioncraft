#include "visioncraft/model.h"
#include "visioncraft/viewpoint.h"
#include "visioncraft/visibility_manager.h"
#include "visioncraft/visualizer.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>

// Function to generate random positions around a sphere
std::vector<Eigen::Vector3d> generateRandomPositions(double radius, int count) {
    std::vector<Eigen::Vector3d> positions;
    for (int i = 0; i < count; ++i) {
        double theta = ((double) rand() / RAND_MAX) * 2 * M_PI;
        double phi = ((double) rand() / RAND_MAX) * M_PI;
        double x = radius * std::sin(phi) * std::cos(theta);
        double y = radius * std::sin(phi) * std::sin(theta);
        double z = radius * std::cos(phi);
        positions.emplace_back(x, y, z);
    }
    return positions;
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

    // Greedy algorithm setup
    double targetCoverage = 0.998;
    double achievedCoverage = visibilityManager->getCoverageScore();
    std::cout << "Initial coverage score: " << achievedCoverage << std::endl;
    std::vector<std::shared_ptr<visioncraft::Viewpoint>> selectedViewpoints;

    int iteration = 0;
    while (achievedCoverage < targetCoverage) {
        std::cout << "\nIteration " << iteration + 1 << " - Starting greedy algorithm loop." << std::endl;

        auto randomPositions = generateRandomPositions(400.0, 10);
        std::cout << "Random positions generated." << std::endl;

        std::shared_ptr<visioncraft::Viewpoint> bestViewpoint = nullptr;
        double bestCoverageIncrement = 0.0;

        // Evaluate each random viewpoint
        for (size_t i = 0; i < randomPositions.size(); ++i) {
            const auto& position = randomPositions[i];
            std::cout << "Evaluating viewpoint " << i + 1 << "/" << randomPositions.size() << " at position: " << position.transpose() << std::endl;

            auto viewpoint = std::make_shared<visioncraft::Viewpoint>(position, Eigen::Vector3d(0.0, 0.0, 0.0));
            viewpoint->setDownsampleFactor(8.0);

            // Track the viewpoint in the visibility manager
            std::cout << "Tracking viewpoint in visibility manager..." << std::endl;
            visibilityManager->trackViewpoint(viewpoint);
            
            std::cout << "Performing raycasting on GPU..." << std::endl;
            viewpoint->performRaycastingOnGPU(model);

            // Calculate the novel coverage score for the current viewpoint
            double coverageIncrement = visibilityManager->computeNovelCoverageScore(viewpoint);
            std::cout << "Coverage increment for this viewpoint: " << coverageIncrement << std::endl;

            // Untrack the viewpoint after evaluation
            visibilityManager->untrackViewpoint(viewpoint);
            std::cout << "Viewpoint untracked." << std::endl;

            if (coverageIncrement > bestCoverageIncrement) {
                bestCoverageIncrement = coverageIncrement;
                bestViewpoint = viewpoint;
                std::cout << "New best viewpoint found with increment: " << bestCoverageIncrement << std::endl;
            }
        }

        // If a suitable viewpoint is found, track it and update coverage
        if (bestViewpoint) {
            std::cout << "Tracking the best viewpoint with coverage increment: " << bestCoverageIncrement << std::endl;
            visibilityManager->trackViewpoint(bestViewpoint);
            std::cout << "Performing raycasting on GPU for the best viewpoint..." << std::endl;
            bestViewpoint->performRaycastingOnGPU(model);
            
            achievedCoverage += bestCoverageIncrement;
            selectedViewpoints.push_back(bestViewpoint);

            std::cout << "Coverage updated: " << achievedCoverage << " / " << targetCoverage << std::endl;
        } else {
            std::cout << "No suitable viewpoint found in this iteration." << std::endl;
        }
        std::cout << "End of iteration " << iteration + 1 << ".\n" << std::endl;
        
        iteration++;
    }

    // Print final results
    std::cout << "\nFinal Coverage Score: " << visibilityManager->getCoverageScore() << std::endl;
    std::cout << "Total Viewpoints Selected: " << selectedViewpoints.size() << std::endl;

    // Render the selected viewpoints and visible voxels
    Eigen::Vector3d baseColor(1.0, 1.0, 1.0);
    Eigen::Vector3d propertyColor(0.0, 1.0, 0.0);
    visualizer.addVoxelMapProperty(model, "visibility", baseColor, propertyColor);

    for (const auto& viewpoint : selectedViewpoints) {
        visualizer.addViewpoint(*viewpoint, true, true);
    }
    visualizer.showGPUVoxelGrid(model, Eigen::Vector3d(0.0, 0.0, 1.0));
    visualizer.render();
    return 0;
}
