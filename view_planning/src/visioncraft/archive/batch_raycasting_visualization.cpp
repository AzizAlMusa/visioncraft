#include "visioncraft/visualizer.h"
#include "visioncraft/viewpoint.h"
#include <Eigen/Dense>
#include <iostream>

int main() {
    // Initialize the visualizer
    visioncraft::Visualizer visualizer;

    // Set the window name and background color
    visualizer.initializeWindow("3D View");
    visualizer.setBackgroundColor(Eigen::Vector3d(0.0, 0.0, 0.0));

    // Create a viewpoint at (400, 0, 0)
    Eigen::Vector3d position(400, 0, 0);
    Eigen::Vector3d lookAt(0, 0, 0); // Looking at the origin
    visioncraft::Viewpoint viewpoint(position, lookAt);
    // Set a downsample factor if needed
    viewpoint.setDownsampleFactor(32.0);

    // Call the function to visualize the rays
    visualizer.showRaysParallel(viewpoint);
    visualizer.addViewpoint(viewpoint, true, true); 
    // Render the visualization
    visualizer.render();

    return 0;
}
