#include <iostream>
#include <fstream>
#include "visioncraft/model_loader.h"
#include "visioncraft/viewpoint.h"
#include <octomap/OcTreeKey.h>
#include <octomap/octomap_types.h>




int main() {
    // Step 1: Create an instance of the ModelLoader class
    visioncraft::ModelLoader model_loader;

    // Load the 3D mesh from the specified path
    std::string file_path = "/home/abdulaziz/ros_workspace/metrology/src/add_post_pro/view_planning/models/bell_scaled.ply";
    if (!model_loader.loadMesh(file_path)) {
        std::cerr << "Failed to load mesh." << std::endl;
        return -1;
    }

    // Step 2: Generate a point cloud with 10,000 samples
    if (!model_loader.generatePointCloud(10000)) {
        std::cerr << "Failed to generate point cloud." << std::endl;
        return -1;
    }

    // Step 3: Generate the exploration map with 32 cells per side
    if (!model_loader.generateExplorationMap(32, model_loader.getMinBound(), model_loader.getMaxBound())) {
        std::cerr << "Failed to generate exploration map." << std::endl;
        return -1;
    }

        // Retrieve the exploration map
    auto exploration_map = model_loader.getExplorationMap();
    if (!exploration_map) {
        std::cerr << "Failed to retrieve Exploration Map." << std::endl;
        return -1;
    }

    // Get and print initial bounds of the exploration map
    double min_x, min_y, min_z, max_x, max_y, max_z;
    exploration_map->getMetricMin(min_x, min_y, min_z);
    exploration_map->getMetricMax(max_x, max_y, max_z);
    std::cout << "Initial exploration map bounds: (" << min_x << ", " << min_y << ", " << min_z
              << ") to (" << max_x << ", " << max_y << ", " << max_z << ")" << std::endl;

    // Step 4: Get the resolution of the exploration map
    double exploration_map_resolution = model_loader.getExplorationMapResolution();
    std::cout << "Exploration map resolution: " << exploration_map_resolution << std::endl;

    // Step 5: Generate an OctoMap using the exploration map resolution
    if (!model_loader.generateOctoMap(exploration_map_resolution)) {
        std::cerr << "Failed to generate OctoMap." << std::endl;
        return -1;
    }

    // Retrieve the generated OctoMap
    auto octomap = model_loader.getOctomap();
    if (!octomap) {
        std::cerr << "Failed to retrieve OctoMap." << std::endl;
        return -1;
    }



    // Step 6: Create a viewpoint at (200, 200, 200) looking at (0, 0, 0)
    Eigen::Vector3d viewpoint_position(200, 200, 200);
    Eigen::Vector3d look_at_point(0, 0, 0);
    visioncraft::Viewpoint viewpoint(viewpoint_position, look_at_point);
    viewpoint.setDownsampleFactor(8.0);  // Downsample the point cloud by a factor of 16

    // Generate rays from the viewpoint
    auto rays = viewpoint.generateRays();
    std::cout << "Number of rays generated: " << rays.size() << std::endl;

    // Step 7: Perform raycasting and update the exploration map
    int hit_count = 0;
    int free_voxels_updated = 0;

    for (const auto& ray_end : rays) {
        octomap::point3d ray_origin(viewpoint_position.x(), viewpoint_position.y(), viewpoint_position.z());
        octomap::point3d ray_target(ray_end.x(), ray_end.y(), ray_end.z());
        octomap::point3d hit_point;

        // Compute keys along the ray from the viewpoint to the endpoint
        octomap::KeyRay keyRay;
        if (octomap->computeRayKeys(ray_origin, ray_target, keyRay)) {
            for (const auto& key : keyRay) {
                octomap::point3d voxel_center = octomap->keyToCoord(key);
                // Ensure voxel is within the bounds of the exploration map
                if (voxel_center.x() >= min_x && voxel_center.x() <= max_x &&
                    voxel_center.y() >= min_y && voxel_center.y() <= max_y &&
                    voxel_center.z() >= min_z && voxel_center.z() <= max_z) {

                    // Search for the node corresponding to the key in the octomap
                    auto node = exploration_map->search(key);
                    auto octomap_node = octomap->search(key);
                    // std::cout << "Node occupancy: " << node->getOccupancy() << " Node value: " << node->getValue()  << std::endl;
                  
                   
                    if (node && octomap_node){
                        node->setLogOdds(1.0);
                        hit_count++;
                        break;
                    } else {
                        node->setLogOdds(-1.0);
                        free_voxels_updated++;
                    }
                }
            }
        }
    }

    std::cout << "Raycasting complete. Total rays: " << rays.size() 
              << ", Hits: " << hit_count 
              << ", Free voxels updated: " << free_voxels_updated << std::endl;

    // Check and print the bounds of the exploration map after raycasting
    exploration_map->getMetricMin(min_x, min_y, min_z);
    exploration_map->getMetricMax(max_x, max_y, max_z);
    std::cout << "Post-raycasting exploration map bounds: (" << min_x << ", " << min_y << ", " << min_z
              << ") to (" << max_x << ", " << max_y << ", " << max_z << ")" << std::endl;

    // Step 8: Save the exploration map leaf nodes to a CSV file
    std::string csv_path = "/home/abdulaziz/ros_workspace/metrology/src/add_post_pro/view_planning/data/exploration_map.csv";
    std::ofstream csv_file(csv_path);
    csv_file << "x,y,z,value\n";  // CSV header

    // Path for the space-delimited text file
    std::string txt_path = "/home/abdulaziz/ros_workspace/metrology/src/add_post_pro/view_planning/data/exploration_map.txt";
    std::ofstream txt_file(txt_path);
    double scale_factor =  0.025 / exploration_map_resolution ;

    // Iterate only over the leaf nodes at the correct level (32x32x32 grid)
    int leaf_count = 0;

    for (auto it = exploration_map->begin_leafs(); it != exploration_map->end_leafs(); ++it) {
        if (it.getDepth() == exploration_map->getTreeDepth()) {
            octomap::point3d coord = it.getCoordinate();
            double value;
      
            if (it->getOccupancy() == 0.5) {
                value = 0.5;  // Unknown
            } else if (exploration_map->isNodeOccupied(*it)) {
                value = 1.0;  // Occupied
            } else {
                value = 0.0;  // Empty
            }

            csv_file << coord.x() << "," << coord.y() << "," << coord.z() << "," << value << "\n";

            // Write to space-delimited text file
            // Scale down the coordinates
            double scaled_x = coord.x() * scale_factor;
            double scaled_y = coord.y() * scale_factor;
            double scaled_z = coord.z() * scale_factor;
            txt_file << scaled_x << " " << scaled_y << " " << scaled_z << " " << value << "\n";

            ++leaf_count;
        }
    }

    csv_file.close();
    txt_file.close();

    std::cout << "Exploration map data saved to " << csv_path << ". Number of elements: " << leaf_count << std::endl;

    // Check if the count matches 32,768
    if (leaf_count == 32768) {
        std::cout << "CSV writing complete: Correct number of leaf nodes written." << std::endl;
    } else {
        std::cerr << "CSV writing complete: Incorrect number of leaf nodes written." << std::endl;
    }

    return 0;
}
