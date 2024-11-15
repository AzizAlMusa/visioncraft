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

// Function to warm up the raycasting system (no timing or metrics collection)
void warmUpRaycasting(visioncraft::Model &model, visioncraft::Visualizer &visualizer) {
    std::cout << "Warming up..." << std::endl;
    
    // Fixed viewpoint at (400, 0, 0) looking at the origin (0, 0, 0)
    Eigen::Vector3d position(400.0, 0.0, 0.0);
    auto viewpoint = std::make_shared<visioncraft::Viewpoint>(position, Eigen::Vector3d(0.0, 0.0, 0.0));
    viewpoint->setDownsampleFactor(1.0);  // No downsampling
    viewpoint->setNearPlane(300);
    viewpoint->setFarPlane(900);

    // Initialize the visibility manager
    auto visibilityManager = std::make_shared<visioncraft::VisibilityManager>(model);

    // Perform a single warm-up iteration (without measuring time)
    viewpoint->setResolution(500, 500); // Middle resolution for warm-up
    visibilityManager->trackViewpoint(viewpoint);
    viewpoint->performRaycastingOnGPU(model);  // Perform raycasting on GPU
    visibilityManager->untrackViewpoint(viewpoint);

    std::cout << "Warm-up complete." << std::endl;
}

// Function to test the raycasting speed for varying x positions
void testGPURaycastingSpeed(visioncraft::Model &model, visioncraft::Visualizer &visualizer) {
    // Print CSV header for raycasting time data
    std::cout << "Position (X),Raycasting Time (ms)" << std::endl;

    // Iterate over x positions from 100 to 800 with a step size of 100
    for (int x_pos = 70; x_pos <= 870; x_pos += 50) {
        // Create a new viewpoint for each x position (keeping y and z fixed)
        Eigen::Vector3d position(x_pos, 0.0, 0.0);
        auto viewpoint = std::make_shared<visioncraft::Viewpoint>(position, Eigen::Vector3d(0.0, 0.0, 0.0));
        viewpoint->setDownsampleFactor(1.0);  // No downsampling
        //print resolution total pixel
        std::cout << " Resolution total pixel: " << viewpoint->getDownsampledResolution().first * viewpoint->getDownsampledResolution().second << std::endl;

        viewpoint->setNearPlane(10);
        viewpoint->setFarPlane(900);

        // Set the default resolution for the viewpoint
        viewpoint->setResolution(500, 500);  // Default resolution

        // Initialize the visibility manager
        auto visibilityManager = std::make_shared<visioncraft::VisibilityManager>(model);

        // Start the timer before raycasting
        auto start = std::chrono::high_resolution_clock::now();

        // Perform raycasting on the GPU for the current viewpoint position
        visibilityManager->trackViewpoint(viewpoint);
        viewpoint->performRaycastingOnGPU(model);  // Perform raycasting on GPU

        // Stop the timer after raycasting
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;  // In milliseconds

        // Output the results in CSV format
        std::cout << x_pos << "," << duration.count() << std::endl;

        // Add overlay text to show the time
        std::ostringstream stream;
        stream << "X Position: " << x_pos << "\nTime: " << duration.count() << " ms";
        visualizer.addOverlayText(stream.str(), 0.05, 0.05, 18, Eigen::Vector3d(1.0, 1.0, 1.0));

        // Add the viewpoint to the visualizer and update the render
        visualizer.addViewpoint(*viewpoint, true, true);
        visualizer.renderStep();

        // Clean up after raycasting
        visibilityManager->untrackViewpoint(viewpoint);

        // Optionally, add a small delay to control the speed of iteration
        std::this_thread::sleep_for(std::chrono::milliseconds(500)); // 500 ms delay between tests
    }
}

int main() {
    srand(time(nullptr));

    // Initialize the visualizer
    visioncraft::Visualizer visualizer;
    visualizer.setBackgroundColor(Eigen::Vector3d(0.0, 0.0, 0.0));

    // Load the model
    visioncraft::Model model;
    model.loadModel("../models/cube.ply", 20000);

    visualizer.addVoxelMap(model);

    // Warm-up phase to avoid high duration in the first iteration
    warmUpRaycasting(model, visualizer);

    // Call the function to test raycasting speed for varying x positions
    testGPURaycastingSpeed(model, visualizer);

    visualizer.render();
    return 0;
}
