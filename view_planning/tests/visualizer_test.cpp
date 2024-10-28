#include "visioncraft/visualizer.h"
#include "visioncraft/model_loader.h"
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


void runModelLoaderMetaVoxelMapTests() {
    visioncraft::ModelLoader model_loader;

    // Load a test mesh and generate the surface shell octomap (mocking actual data for this test)
    std::string test_file_path = "../models/cube.ply"; // Replace with a valid test path
    bool loaded = model_loader.loadModel(test_file_path, 50000);

    if (!loaded) {
        std::cerr << "Failed to load test model and initialize structures." << std::endl;
        return;
    }

    // Generate the meta voxel map from the surface shell octomap
    if (!model_loader.generateMetaVoxelMap()) {
        std::cerr << "Failed to generate meta voxel map." << std::endl;
        return;
    }

    // Select a sample key (this assumes the map has been populated)
    auto sample_key = model_loader.getSurfaceShellOctomap()->begin_leafs().getKey();
    visioncraft::MetaVoxel* meta_voxel = model_loader.getMetaVoxel(sample_key);

    if (meta_voxel) {
        // Output initial state of the MetaVoxel
        std::cout << "Initial MetaVoxel at key (" << sample_key.k[0] << ", " << sample_key.k[1] << ", " << sample_key.k[2] << ")" << std::endl;
        std::cout << "  Position: " << meta_voxel->getPosition().transpose() << std::endl;
        std::cout << "  Occupancy: " << meta_voxel->getOccupancy() << std::endl;

        // Update occupancy and check result
        model_loader.updateMetaVoxelOccupancy(sample_key, 0.8f);
        std::cout << "Updated occupancy to 0.8 for MetaVoxel." << std::endl;
        std::cout << "  New Occupancy: " << meta_voxel->getOccupancy() << std::endl;

        // Set the "temperature" property and then retrieve it
        model_loader.setMetaVoxelProperty(sample_key, "temperature", 22.5f);
        float temperature = boost::get<float>(model_loader.getMetaVoxelProperty(sample_key, "temperature"));
        std::cout << "Set custom property 'temperature' to 22.5." << std::endl;
        std::cout << "  Retrieved Temperature: " << temperature << std::endl;

        // Set another custom property (e.g., "pressure") and retrieve it
        model_loader.setMetaVoxelProperty(sample_key, "pressure", 101.3f);
        float pressure = boost::get<float>(model_loader.getMetaVoxelProperty(sample_key, "pressure"));
        std::cout << "Set custom property 'pressure' to 101.3." << std::endl;
        std::cout << "  Retrieved Pressure: " << pressure << std::endl;

        // Attempt to retrieve a non-existing property to test error handling
        try {
            auto nonexistent = model_loader.getMetaVoxelProperty(sample_key, "nonexistent_property");
        } catch (const std::runtime_error& e) {
            std::cerr << "Expected error for non-existing property: " << e.what() << std::endl;
        }
    } else {
        std::cerr << "Failed to retrieve MetaVoxel for the provided key." << std::endl;
    }
}


int main() {

    runModelLoaderMetaVoxelMapTests();



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
