#include "visioncraft/visualizer.h"
#include "visioncraft/model.h"
#include "visioncraft/viewpoint.h"
#include "visioncraft/meta_voxel.h"
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

#include <random>

void updateVoxelMapVisibility(
    const std::unordered_map<octomap::OcTreeKey, bool, octomap::OcTreeKey::KeyHash>& octree_hits, 
    visioncraft::Model& model) {
    
    for (const auto& pair : octree_hits) {
        const octomap::OcTreeKey& key = pair.first;
        bool hit = pair.second;

        if (!hit) continue;  // Skip if voxel wasn't hit

        // Access the MetaVoxel corresponding to the OctoMap key
        visioncraft::MetaVoxel* meta_voxel = model.getVoxel(key);

        if (meta_voxel) {
            // Verify and update the visibility property
            int visibility = boost::get<int>(meta_voxel->getProperty("visibility")) + 1;
            meta_voxel->setProperty("visibility", visibility);
        } else {
            std::cerr << "No MetaVoxel found for key (" 
                      << key.k[0] << ", " << key.k[1] << ", " << key.k[2] << ")" << std::endl;
        }
    }
}


// Function to generate random positions at a given radius
std::vector<Eigen::Vector3d> generateRandomPositions(int n, double radius) {
    std::vector<Eigen::Vector3d> positions;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> azimuth_dist(0, 2 * M_PI); // Full circle for azimuth
    std::uniform_real_distribution<> elevation_dist(0, M_PI);   // Half-circle for elevation

    for (int i = 0; i < n; ++i) {
        double azimuth = azimuth_dist(gen);
        double elevation = elevation_dist(gen);

        // Convert spherical to Cartesian coordinates
        double x = radius * std::sin(elevation) * std::cos(azimuth);
        double y = radius * std::sin(elevation) * std::sin(azimuth);
        double z = radius * std::cos(elevation);

        positions.emplace_back(x, y, z);
    }
    return positions;
}


int main() {



    // Initialize Visualizer
    visioncraft::Visualizer visualizer;

    // Set the window name and background color
    visualizer.initializeWindow("3D View");
    visualizer.setBackgroundColor(Eigen::Vector3d(0.0, 0.0, 0.0));

    // // Load model (you can customize this part according to your Model implementation)
    visioncraft::Model model;
    model.loadModel("../models/cube.ply", 50000);  // Replace with actual file path

    // cast the 0 as int 
    model.addVoxelProperty("visibility", 0);  // Initialize visibility property for all voxels

    // // Add the octomap to the visualizer
    // visualizer.addOctomap(model, Eigen::Vector3d(1.0, 1.0, 1.0));

    // Create multiple viewpoints
    std::vector<Eigen::Vector3d> positions = {
        Eigen::Vector3d(400, 0, 0),  // +X axis
        Eigen::Vector3d(-400, 0, 0), // -X axis
        Eigen::Vector3d(0, 400, 0),  // +Y axis
        Eigen::Vector3d(0, -400, 0), // -Y axis
        Eigen::Vector3d(0, 0, 400),  // +Z axis
        Eigen::Vector3d(0, 0, -400)  // -Z axis
    };

    // Generate n random positions at radius 400
    int n = 1; // Number of random positions
    double radius = 400.0;
    // std::vector<Eigen::Vector3d> positions = generateRandomPositions(n, radius);

    Eigen::Vector3d lookAt(0.0, 0.0, 0.0); // All viewpoints will look at the origin

    // // Initialize voxel count
    // unsigned int total_voxels = 0;

    // // Get the octomap from the model loader
    // auto octomap = model.getOctomap();

    // // Iterate through all the leaf nodes in the octomap to get the total voxel count
    // for (octomap::ColorOcTree::leaf_iterator it = octomap->begin_leafs(), end = octomap->end_leafs(); it != end; ++it) {
    //     total_voxels++;
    // }

    // // Output the total number of voxels
    // std::cout << "Total number of voxels: " << total_voxels << std::endl;

    // // To store hit voxels across all viewpoints
    // std::unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash> unique_hits;
    
   int counter = 0;
    // Perform raycasting for each viewpoint
    for (const auto& position : positions) {
        visioncraft::Viewpoint viewpoint(position, lookAt);
        viewpoint.setDownsampleFactor(8);

        //After initializing each viewpoint in Python
        std::cout << "Orientation Matrix after setLookAt in C++:\n" << viewpoint.getOrientationMatrix() << std::endl;



  
        auto start = std::chrono::high_resolution_clock::now();
 
        auto hit_results = viewpoint.performRaycastingOnGPU(model);
        // auto hit_results = viewpoint.performRaycasting(model, true);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "Raycasting for viewpoint at position (" << position.x() << ", " << position.y() << ", " << position.z() << ") took: " << elapsed.count() << " ms" << std::endl;
        
        // model.updateVoxelGridFromHits(hit_results);
        // model.updateOctomapWithHits(hit_results);
        // Update visibility in the MetaVoxelMap for each hit voxel
        updateVoxelMapVisibility(hit_results, model);

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
        // if (counter == 2) visualizer.showRays(viewpoint, Eigen::Vector3d(1.0, 0.0, 0.0));  // Red rays
        

        counter++;
        // // Perform raycasting for this viewpoint
        // auto start = std::chrono::high_resolution_clock::now();
        // auto hit_results = viewpoint.performRaycasting(model.getOctomap(), true);
       
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
        // visualizer.showRayVoxels(viewpoint, model.getOctomap(), Eigen::Vector3d(1.0, 0.0, 0.0));  // Red voxels


    }

    // Get the total voxels hit by all viewpoints and print it
    // std::cout << "Total unique voxels hit by all viewpoints: " << unique_hits.size() << std::endl;




    // Visualize raycasting results
    // visualizer.showViewpointHits(model.getOctomap());
    // visualizer.addOctomap(model);
    Eigen::Vector3d baseColor(1.0, 1.0, 1.0);  // Example: Red color for the voxels
    Eigen::Vector3d propertyColor(0.0, 1.0, 0.0);  // Example: Green color for the voxels
    // visualizer.addVoxelMap(model, voxelColor);
    visualizer.addVoxelMapProperty(model, "visibility", baseColor, propertyColor, 0, 2); // Scale 0 to 10 as example

    // visualizer.showGPUVoxelGrid(model, voxelColor);
    // Start the rendering loop
    visualizer.render();

    return 0;
}
