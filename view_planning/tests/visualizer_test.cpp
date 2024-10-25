#include "visioncraft/visualizer.h"
#include "visioncraft/model_loader.h"
#include "visioncraft/viewpoint.h"
#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkPolyDataMapper.h>
#include <vtkAxesActor.h>
#include <vtkCubeSource.h>
#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <unordered_set> // Use unordered_set to track unique voxels
#include <cstdlib>
#include <cuda_runtime.h> // For CUDA functions

#include <vtkAutoInit.h>

VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkInteractionStyle);



int main() {

    // Query and print CUDA device properties

    // Initialize Visualizer
    visioncraft::Visualizer visualizer;

    // Set the window name and background color
    visualizer.initializeWindow("3D View");
    visualizer.setBackgroundColor(Eigen::Vector3d(0.0, 0.0, 0.0));

    // // Load model (you can customize this part according to your ModelLoader implementation)
    visioncraft::ModelLoader modelLoader;
    modelLoader.loadModel("../models/gorilla.ply", 50000);  // Replace with actual file path

    // // Add the octomap to the visualizer
    // visualizer.addOctomap(modelLoader, Eigen::Vector3d(1.0, 1.0, 1.0));

    // Create multiple viewpoints
    std::vector<Eigen::Vector3d> positions = {
        Eigen::Vector3d(400, 0, 0),  // +X axis
        Eigen::Vector3d(-400, 0, 0), // -X axis
        // Eigen::Vector3d(0, 400, 0),  // +Y axis
        // Eigen::Vector3d(0, -400, 0), // -Y axis
        // Eigen::Vector3d(0, 0, 400),  // +Z axis
        // Eigen::Vector3d(0, 0, -400)  // -Z axis
    };

    Eigen::Vector3d lookAt(0.0, 0.0, 0.0); // All viewpoints will look at the origin

    // // Initialize voxel count
    // unsigned int total_voxels = 0;

    // // Get the octomap from the model loader
    // auto octomap = modelLoader.getOctomap();

    // // Iterate through all the leaf nodes in the octomap to get the total voxel count
    // for (octomap::ColorOcTree::leaf_iterator it = octomap->begin_leafs(), end = octomap->end_leafs(); it != end; ++it) {
    //     total_voxels++;
    // }

    // // Output the total number of voxels
    // std::cout << "Total number of voxels: " << total_voxels << std::endl;

    // // To store hit voxels across all viewpoints
    // std::unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash> unique_hits;

    // Perform raycasting for each viewpoint
    for (const auto& position : positions) {
        visioncraft::Viewpoint viewpoint(position, lookAt);
        viewpoint.setDownsampleFactor(8);

  
        auto start = std::chrono::high_resolution_clock::now();
 
        auto hit_results = viewpoint.performRaycastingOnGPU(modelLoader);
        // auto hit_results = viewpoint.performRaycasting(modelLoader, true);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "Raycasting for viewpoint at position (" << position.x() << ", " << position.y() << ", " << position.z() << ") took: " << elapsed.count() << " ms" << std::endl;
        
        // modelLoader.updateVoxelGridFromHits(hit_results);
        modelLoader.updateOctomapWithHits(hit_results);

        // Compare the results
        // for (int i = 0; i < cpu_rays.size(); ++i) {
        //     if ((cpu_rays[i] - gpu_rays[i]).norm() > 1e-3) {
        //         std::cout << "Difference in ray " << i << ": CPU [" 
        //                 << cpu_rays[i].transpose() << "] vs GPU [" 
        //                 << gpu_rays[i].transpose() << "]" << std::endl;
        //     }
        // }


        // Add the viewpoint to the visualizer
        visualizer.addViewpoint(viewpoint, true, true);

        // // Perform raycasting for this viewpoint
        // auto start = std::chrono::high_resolution_clock::now();
        // auto hit_results = viewpoint.performRaycasting(modelLoader.getOctomap(), true);
       
        // auto gpu_rays = viewpoint.performRaycastingOnGPU();  // GPU raycasting
        // auto end = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double, std::milli> elapsed = end - start;
        // std::cout << "Raycasting for viewpoint at position (" << position.x() << ", " << position.y() << ", " << position.z() << ") took: " << elapsed.count() << " ms" << std::endl;

        // // Get the hit results by each viewpoint and print it
        // unsigned int hit_count = 0;
        // for (const auto& hit_result : hit_results) {
        //     const auto& key = hit_result.first; // Access the key
        //     const auto& hit = hit_result.second; // Access the hit status

        //     if (hit) {
        //         hit_count++;
        //         // Add hit voxel key to unique_hits set
        //         unique_hits.insert(key);
        //     }
        // }
        // std::cout << "Viewpoint at position (" << position.x() << ", " << position.y() << ", " << position.z() << ") hit " << hit_count << " voxels.\n";
        
        // Show the rays as voxels
        // visualizer.showRayVoxels(viewpoint, modelLoader.getOctomap(), Eigen::Vector3d(1.0, 0.0, 0.0));  // Red voxels


    }

    // Get the total voxels hit by all viewpoints and print it
    // std::cout << "Total unique voxels hit by all viewpoints: " << unique_hits.size() << std::endl;

    
    
    // Visualize raycasting results
    // visualizer.showViewpointHits(modelLoader.getOctomap());
    visualizer.addOctomap(modelLoader);
    Eigen::Vector3d voxelColor(1.0, 0.0, 0.0);  // Example: Red color for the voxels
    visualizer.showGPUVoxelGrid(modelLoader, voxelColor);
    // Start the rendering loop
    visualizer.render();

    return 0;
}
