#include "visioncraft/model.h"
#include "visioncraft/viewpoint.h"
#include "visioncraft/visibility_manager.h"
#include "visioncraft/visualizer.h"
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <memory>
#include <chrono>
#include <thread>
#include <vector>
#include <iomanip>  // For setting decimal precision

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
    srand(time(nullptr));

    // Initialize the visualizer
    visioncraft::Visualizer visualizer;
    visualizer.setBackgroundColor(Eigen::Vector3d(0.0, 0.0, 0.0));

    // Load the model
    visioncraft::Model model;
    model.loadModel("../models/gorilla.ply", 20000);

    // Initialize the visibility manager
    auto visibilityManager = std::make_shared<visioncraft::VisibilityManager>(model);

    // Camera rotation parameters (fixed camera view)
    double radius = 600.0; // Distance of the camera from the object
    double elevation = 30.0 * M_PI / 180.0;  // Elevation angle in radians (30 degrees)
    double angle = -45.0 * M_PI / 180.0;    // Start angle in radians

    // Calculate the camera position in 3D space
    double x = radius * std::cos(angle) * std::cos(elevation);
    double y = radius * std::sin(angle) * std::cos(elevation);
    double z = radius * std::sin(elevation);

    // Set the camera position and focal point
    visualizer.getRenderer()->GetActiveCamera()->SetPosition(x, y, z);
    visualizer.getRenderer()->GetActiveCamera()->SetFocalPoint(0.0, 0.0, 0.0);
    visualizer.getRenderer()->GetActiveCamera()->SetViewUp(0.0, 0.0, 1.0);  // Ensure the "up" direction is consistent

    // Declare total loop time variable
    auto totalLoopStart = std::chrono::high_resolution_clock::now();  // Start tracking cumulative loop time
    double cumulativeComputationTime = 0.0;  // To accumulate the total computation time

    // Loop for generating random viewpoints and performing raycasting
    for (int iteration = 0; iteration < 500; ++iteration) { // Adjust number of iterations as needed
        // Generate random position and create a viewpoint
        Eigen::Vector3d position = generateRandomPositions(300, 1)[0]; // Get one random position
        auto viewpoint = std::make_shared<visioncraft::Viewpoint>(position, Eigen::Vector3d(0.0, 0.0, 0.0));
        viewpoint->setDownsampleFactor(8.0);
        viewpoint->setNearPlane(100);
        viewpoint->setFarPlane(300);

        // Perform raycasting (no threading, just sequential)
        visibilityManager->trackViewpoint(viewpoint);

        // Start the timer before raycasting
        auto start = std::chrono::high_resolution_clock::now();

        // viewpoint->performRaycasting(model, true);
        viewpoint->performRaycastingOnGPU(model);    // Perform raycasting on GPU

        // Stop the timer after raycasting
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;  // In milliseconds

        // Add the duration of the current iteration to the cumulative computation time
        cumulativeComputationTime += duration.count() / 1000.0;  // Convert ms to seconds

        // Compute and display the cumulative loop time (total time elapsed since the start of the loop)
        auto totalLoopEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> totalLoopDuration = totalLoopEnd - totalLoopStart;  // Total loop time in seconds

        // Prepare the message
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(2);
        stream << "Viewpoints evaluated   =   " << (iteration + 1) << 
                "\n\nComputation time         =   " << static_cast<int>(duration.count()) << " ms" <<
                "\n\nSpeed (evaluations/s)   =   " << (iteration + 1) / cumulativeComputationTime <<
                "\n\nTime                                 =   " << totalLoopDuration.count() << " s" ;

        // Overlay the iteration count, computation time, and cumulative loop time on the visualizer
        visualizer.removeOverlayTexts(); // Remove previous overlay texts
        visualizer.addOverlayText(stream.str(), 0.05, 0.05, 18, Eigen::Vector3d(1.0, 1.0, 1.0)); // Display text in white color

        // Add the viewpoint to the visualizer and update the render
        visualizer.addViewpoint(*viewpoint, true, true);
        visualizer.addVoxelMapProperty(model, "visibility", Eigen::Vector3d(1.0, 1.0, 1.0), Eigen::Vector3d(0.0, 1.0, 0.0)); // Add visibility map

        // Render the updated scene
        visualizer.renderStep();

        // Clean up after rendering the frame
        visibilityManager->untrackViewpoint(viewpoint);
        visualizer.removeViewpoints(); // Remove previous viewpoints
        visualizer.removeVoxelMapProperty(); // Remove previous voxel map property

        // Optionally, add a small delay to control the speed of iteration
        std::this_thread::sleep_for(std::chrono::milliseconds(1)); // 1 ms delay between iterations
    }

    return 0;
}
