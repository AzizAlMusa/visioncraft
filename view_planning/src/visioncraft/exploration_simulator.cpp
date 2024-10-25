#include "visioncraft/exploration_simulator.h"
#include <iostream>

namespace visioncraft {

ExplorationSimulator::ExplorationSimulator() : scaling_factor_(1.0) {}

bool ExplorationSimulator::loadModel(const std::string& model_file, int num_points, int grid_resolution) {
    ModelLoader model_loader;

    if (!model_loader.loadExplorationModel(model_file, num_points, grid_resolution)) {
        std::cerr << "Failed to load mesh." << std::endl;
        return false;
    }

    exploration_map_ = model_loader.getExplorationMap();
    octomap_ = model_loader.getOctomap();
    surface_shell_map_ = model_loader.getSurfaceShellOctomap();

    scaling_factor_ = 0.025 / exploration_map_->getResolution();

    return true;
}

void ExplorationSimulator::setViewpoints(const std::vector<Viewpoint>& viewpoints) {
    viewpoints_ = viewpoints;
}

void ExplorationSimulator::performGuidedRaycasting(Viewpoint& viewpoint) {
    int hit_count = 0;           // Counter for the number of occupied voxels (hits)
    int free_voxels_updated = 0; // Counter for the number of free (empty) voxels updated

    // Generate rays from the viewpoint
    auto rays = viewpoint.generateRays();
    std::cout << "Number of rays generated: " << rays.size() << std::endl;

    // Get the boundaries of the exploration map to ensure updates are within valid voxels
    double min_x, min_y, min_z, max_x, max_y, max_z;
    exploration_map_->getMetricMin(min_x, min_y, min_z);
    exploration_map_->getMetricMax(max_x, max_y, max_z);

    // Iterate over each generated ray
    for (const auto& ray_end : rays) {
        // Define the origin and target points of the ray
        octomap::point3d ray_origin(viewpoint.getPosition().x(), viewpoint.getPosition().y(), viewpoint.getPosition().z());
        octomap::point3d ray_target(ray_end.x(), ray_end.y(), ray_end.z());
        octomap::KeyRay keyRay; // Container for storing the keys (voxels) along the ray path

        // Compute the voxel keys along the ray
        if (octomap_->computeRayKeys(ray_origin, ray_target, keyRay)) {
            // Iterate over each key (voxel) along the ray path
            for (const auto& key : keyRay) {
                octomap::point3d voxel_center = octomap_->keyToCoord(key); // Get the center of the voxel

                // Ensure voxel is within the boundaries of the exploration map
                if (voxel_center.x() >= min_x && voxel_center.x() <= max_x &&
                    voxel_center.y() >= min_y && voxel_center.y() <= max_y &&
                    voxel_center.z() >= min_z && voxel_center.z() <= max_z) {

                    auto node = exploration_map_->search(key);    // Search for the voxel in the exploration map
                    auto octomap_node = octomap_->search(key);    // Search for the voxel in the octomap

                    if (node && octomap_node) {
                        // If the voxel is found in both maps, mark it as occupied
                        node->setLogOdds(1.0);  
                        hit_count++;
                        break; // Stop processing the ray once the first occupied voxel is found
                    } else if (node) {
                        // If the voxel is only found in the exploration map, mark it as free
                        node->setLogOdds(-1.0);  
                        free_voxels_updated++;
                    }
                }
            }
        }
    }

    // Output the results of the raycasting process
    std::cout << "Raycasting complete. Total rays: " << rays.size()
              << ", Hits: " << hit_count
              << ", Free voxels updated: " << free_voxels_updated << std::endl;
}


void ExplorationSimulator::performRaycasting() {
    for (auto& viewpoint : viewpoints_) {
        performGuidedRaycasting(viewpoint);
    }
}


std::vector<std::tuple<double, double, double, double>> ExplorationSimulator::getExplorationMapData() {
    std::vector<std::tuple<double, double, double, double>> data;

    // Calculate the scale factor
    double exploration_map_resolution = exploration_map_->getResolution();
    double scale_factor = 0.025 / exploration_map_resolution;

    int leaf_count = 0; // Counter for the number of leaf nodes

    // Iterate over the leaf nodes in the exploration map
    for (auto it = exploration_map_->begin_leafs(); it != exploration_map_->end_leafs(); ++it) {
        if (it.getDepth() == exploration_map_->getTreeDepth()) { // Ensure processing only at the correct depth
            octomap::point3d coord = it.getCoordinate(); // Get the coordinates of the leaf node
            double value;

            // Determine the occupancy value of the node
            if (it->getOccupancy() == 0.5) {
                value = 0.5;  // Unknown
            } else if (exploration_map_->isNodeOccupied(*it)) {
                value = 1.0;  // Occupied
            } else {
                value = 0.0;  // Empty
            }

            // Scale coordinates
            double scaled_x = coord.x() * scale_factor;
            double scaled_y = coord.y() * scale_factor;
            double scaled_z = coord.z() * scale_factor;

            // Add to data vector
            data.emplace_back(scaled_x, scaled_y, scaled_z, value);

            ++leaf_count;
        }   
    }

    // Output the results of the data extraction process
    std::cout << "Exploration map data extracted. Number of elements: " << leaf_count << std::endl;

    // Check if the number of nodes matches the expected count
    if (leaf_count == 32768) {
        std::cout << "Data extraction complete: Correct number of leaf nodes extracted." << std::endl;
    } else {
        std::cerr << "Data extraction complete: Incorrect number of leaf nodes extracted." << std::endl;
    }

    return data;
}


double ExplorationSimulator::getCoverageScore() {
    int matched_occupied_voxels = 0;
    int total_occupied_voxels_surface = 0;

    for (auto it = surface_shell_map_->begin_leafs(); it != surface_shell_map_->end_leafs(); ++it) {
        if (surface_shell_map_->isNodeOccupied(*it)) {
            total_occupied_voxels_surface++;
            octomap::point3d surface_voxel_center = it.getCoordinate();

            auto exploration_node = exploration_map_->search(surface_voxel_center);
            if (exploration_node && exploration_node->getOccupancy() > 0.5) {
                matched_occupied_voxels++;
            }
        }
    }

    return static_cast<double>(matched_occupied_voxels) / total_occupied_voxels_surface;
}

double ExplorationSimulator::getScalingFactor() const {
    return scaling_factor_;
}

} // namespace visioncraft
