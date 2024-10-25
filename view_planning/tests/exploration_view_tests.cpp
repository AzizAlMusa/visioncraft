#include <iostream>
#include <fstream>
#include "visioncraft/model_loader.h"
#include "visioncraft/viewpoint.h"
#include <octomap/OcTreeKey.h>
#include <octomap/octomap_types.h>

namespace visioncraft {


    /**
     * @brief Read viewpoints from a file where each line contains x, y, z, lookAt_x, lookAt_y, lookAt_z.
     *
     * This function reads a file containing viewpoint data. Each line in the file is expected to have six 
     * space-delimited values representing the position (x, y, z) and the lookAt direction (lookAt_x, lookAt_y, lookAt_z).
     * It constructs a Viewpoint object from these values and returns a vector of these Viewpoint objects.
     *
     * @param filename The path to the file containing the viewpoints.
     * @return std::vector<Viewpoint> A vector containing the Viewpoint objects read from the file.
     */
    std::vector<Viewpoint> readViewpointsFromFile(const std::string& filename) {
        std::vector<Viewpoint> viewpoints;
        std::ifstream file(filename);

        // Check if the file can be opened
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return viewpoints;
        }

        double x, y, z, lookAt_x, lookAt_y, lookAt_z;

        // Read the file line by line
        while (file >> x >> y >> z >> lookAt_x >> lookAt_y >> lookAt_z) {
            // Construct the position vector (x, y, z)
            Eigen::Vector3d position(x, y, z);

            // Construct the lookAt vector (lookAt_x, lookAt_y, lookAt_z)
            Eigen::Vector3d lookAt(lookAt_x, lookAt_y, lookAt_z);

            // Create a Viewpoint object using the position and lookAt vectors
            Viewpoint viewpoint(position, lookAt);

            // Add the created viewpoint to the vector
            viewpoints.push_back(viewpoint);
        }

        // Close the file after reading
        file.close();
        return viewpoints;
    }

    /**
     * @brief Save the scaling factor to a file.
     *
     * This function saves the calculated scaling factor to a specified file.
     * The scaling factor is important for ensuring that the neural network
     * predictions are properly scaled relative to the voxel size used in the
     * exploration map.
     *
     * @param scaling_factor The scaling factor to be saved.
     * @param file_path The path of the file where the scaling factor will be saved.
     */
    void saveScalingFactor(double scaling_factor, const std::string& file_path) {
        std::ofstream file(file_path);  // Open the file at the specified path for writing
        if (file.is_open()) {
            file << scaling_factor;  // Write the scaling factor to the file
            file.close();  // Close the file after writing
            std::cout << "Scaling factor saved to " << file_path << ": " << scaling_factor << std::endl;
        } else {
            std::cerr << "Error opening file to save scaling factor: " << file_path << std::endl;
        }
    }


    /**
     * @brief Perform raycasting from a viewpoint and update the exploration map based on the octomap.
     *
     * This function casts rays from a specified viewpoint, updating the exploration map by marking 
     * the occupied and free voxels according to their presence in the octomap.
     *
     * @param exploration_map The exploration map to be updated.
     * @param octomap The octomap containing the environment data.
     * @param viewpoint The viewpoint from which rays are cast.
     */
    void performGuidedRaycasting(std::shared_ptr<octomap::ColorOcTree>& exploration_map,
                                 std::shared_ptr<octomap::ColorOcTree>& octomap,
                                 Viewpoint& viewpoint) {

        int hit_count = 0;           // Counter for the number of occupied voxels (hits)
        int free_voxels_updated = 0; // Counter for the number of free (empty) voxels updated

        // Generate rays from the viewpoint
        auto rays = viewpoint.generateRays();
        std::cout << "Number of rays generated: " << rays.size() << std::endl;

        // Get the boundaries of the exploration map to ensure updates are within valid voxels
        double min_x, min_y, min_z, max_x, max_y, max_z;
        exploration_map->getMetricMin(min_x, min_y, min_z);
        exploration_map->getMetricMax(max_x, max_y, max_z);

        // Iterate over each generated ray
        for (const auto& ray_end : rays) {
            // Define the origin and target points of the ray
            octomap::point3d ray_origin(viewpoint.getPosition().x(), viewpoint.getPosition().y(), viewpoint.getPosition().z());
            octomap::point3d ray_target(ray_end.x(), ray_end.y(), ray_end.z());
            octomap::KeyRay keyRay; // Container for storing the keys (voxels) along the ray path

            // Compute the voxel keys along the ray
            if (octomap->computeRayKeys(ray_origin, ray_target, keyRay)) {
                // Iterate over each key (voxel) along the ray path
                for (const auto& key : keyRay) {
                    octomap::point3d voxel_center = octomap->keyToCoord(key); // Get the center of the voxel

                    // Ensure voxel is within the boundaries of the exploration map
                    if (voxel_center.x() >= min_x && voxel_center.x() <= max_x &&
                        voxel_center.y() >= min_y && voxel_center.y() <= max_y &&
                        voxel_center.z() >= min_z && voxel_center.z() <= max_z) {

                        auto node = exploration_map->search(key);    // Search for the voxel in the exploration map
                        auto octomap_node = octomap->search(key);    // Search for the voxel in the octomap

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

    /**
     * @brief Perform raycasting for multiple viewpoints and update the exploration map.
     *
     * This function iterates over a vector of viewpoints, performing raycasting for each one
     * and updating the exploration map based on the octomap.
     *
     * @param exploration_map The exploration map to be updated.
     * @param octomap The octomap containing the environment data.
     * @param viewpoints A vector of viewpoints from which rays are cast.
     */
    void performRaycastingForViewpoints(std::shared_ptr<octomap::ColorOcTree>& exploration_map,
                                        std::shared_ptr<octomap::ColorOcTree>& octomap,
                                        std::vector<Viewpoint>& viewpoints) {

        for (auto& viewpoint : viewpoints) {
            // Perform raycasting for each viewpoint
            performGuidedRaycasting(exploration_map, octomap, viewpoint);
        }
    }

    /**
     * @brief Save the exploration map's leaf nodes to CSV and TXT files.
     *
     * This function iterates over the leaf nodes of the exploration map and saves their coordinates 
     * and occupancy values to a CSV file and a space-delimited TXT file. The TXT file scales down 
     * the coordinates based on the resolution.
     *
     * @param exploration_map The exploration map containing the leaf nodes.
     * @param csv_path Path to save the CSV file.
     * @param txt_path Path to save the TXT file.
     */
    void saveExplorationMapToFiles(const std::shared_ptr<octomap::ColorOcTree>& exploration_map,
                                   const std::string& csv_path,
                                   const std::string& txt_path) {

        // Open CSV and TXT files for writing
        std::ofstream csv_file(csv_path);
        std::ofstream txt_file(txt_path);

        // Write CSV header
        csv_file << "x,y,z,value\n";

        // Calculate the scale factor for the TXT file coordinates
        double exploration_map_resolution = exploration_map->getResolution();
        double scale_factor = 0.025 / exploration_map_resolution;

        int leaf_count = 0; // Counter for the number of leaf nodes
        // Iterate over the leaf nodes in the exploration map
        for (auto it = exploration_map->begin_leafs(); it != exploration_map->end_leafs(); ++it) {
            if (it.getDepth() == exploration_map->getTreeDepth()) { // Ensure processing only at the correct depth
                octomap::point3d coord = it.getCoordinate(); // Get the coordinates of the leaf node
                double value;

                // Determine the occupancy value of the node
                if (it->getOccupancy() == 0.5) {
                    value = 0.5;  // Unknown
                } else if (exploration_map->isNodeOccupied(*it)) {
                    value = 1.0;  // Occupied
                } else {
                    value = 0.0;  // Empty
                }

                // Write to CSV file
                csv_file << coord.x() << "," << coord.y() << "," << coord.z() << "," << value << "\n";

                // Scale coordinates and write to TXT file
                double scaled_x = coord.x() * scale_factor;
                double scaled_y = coord.y() * scale_factor;
                double scaled_z = coord.z() * scale_factor;
                txt_file << scaled_x << " " << scaled_y << " " << scaled_z << " " << value << "\n";

                ++leaf_count;
            }   
        }

        // Close the files after writing
        csv_file.close();
        txt_file.close();

        // Output the results of the file writing process
        std::cout << "Exploration map data saved to " << csv_path << ". Number of elements: " << leaf_count << std::endl;

        // Check if the number of written nodes matches the expected count
        if (leaf_count == 32768) {
            std::cout << "CSV writing complete: Correct number of leaf nodes written." << std::endl;
        } else {
            std::cerr << "CSV writing complete: Incorrect number of leaf nodes written." << std::endl;
        }
    }

    /**
     * @brief Compare the exploration map with the surface shell map and calculate the coverage ratio.
     *
     * This function compares the occupied voxels in the exploration map with those in the surface shell map.
     * It counts how many occupied voxels in the exploration map match the occupied voxels in the surface map,
     * and returns the ratio of matched occupied voxels to the total number of occupied voxels in the surface map.
     *
     * @param exploration_map The exploration map to be compared.
     * @param surface_map The surface shell map for comparison.
     * @return double The ratio of matched occupied voxels to the total occupied voxels in the surface map.
     */
    double getCoverageScore(std::shared_ptr<octomap::ColorOcTree>& exploration_map,
                            std::shared_ptr<octomap::ColorOcTree>& surface_map) {

        int matched_occupied_voxels = 0; // Counter for matched occupied voxels
        int total_occupied_voxels_surface = 0; // Counter for total occupied voxels in the surface map

        // Iterate over the surface map's occupied voxels
        for (auto it = surface_map->begin_leafs(); it != surface_map->end_leafs(); ++it) {
            if (surface_map->isNodeOccupied(*it)) {
                total_occupied_voxels_surface++; // Increment total occupied voxels counter
                octomap::point3d surface_voxel_center = it.getCoordinate(); // Get the voxel's coordinates
                
                // Find the corresponding voxel in the exploration map
                auto exploration_node = exploration_map->search(surface_voxel_center);
                if (exploration_node && exploration_node->getOccupancy() > 0.5) {
                    matched_occupied_voxels++; // Increment matched occupied voxels counter
                }
            }
        }

        // Return the ratio of matched occupied voxels to the total occupied voxels in the surface map
        return static_cast<double>(matched_occupied_voxels) / total_occupied_voxels_surface;
    }

}  // namespace visioncraft

int main(int argc, char* argv[]) {

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model_file> <nbv_list_file>" << std::endl;
        return 1;
    }

    std::string model_file = argv[1];
    std::string nbv_list_file = argv[2];


    visioncraft::ModelLoader model_loader;

    // Load the 3D model, generate a point cloud, and create the exploration map
    // std::string file_path = "/home/abdulaziz/ros_workspace/metrology/src/add_post_pro/view_planning/models/gorilla.ply";
    if (!model_loader.loadExplorationModel(model_file, 20000, 32)) {
        std::cerr << "Failed to load mesh." << std::endl;
        return -1;
    }

    // Retrieve the generated exploration map and octomap
    auto exploration_map = model_loader.getExplorationMap();
    auto octomap = model_loader.getOctomap(); 
    double scale_factor = 0.025 / exploration_map->getResolution();
    visioncraft::saveScalingFactor(scale_factor, "../data/scaling_factor.txt");
    // Define the viewpoint's position and target point
    // Create a vector of viewpoints
    // Read viewpoints from file
    std::vector<visioncraft::Viewpoint>  viewpoints = visioncraft::readViewpointsFromFile(nbv_list_file);


    // Set downsample factor for each viewpoint
    for (auto& viewpoint : viewpoints) {
        viewpoint.setDownsampleFactor(4.0);
    }
    // Perform raycasting and update the exploration map
    visioncraft::performRaycastingForViewpoints(exploration_map, octomap, viewpoints);



    // Save the exploration map to CSV and TXT files

    // Obtain the directory of the current cpp file
    std::string cppFilePath = __FILE__;
    std::string cppDirectory = cppFilePath.substr(0, cppFilePath.find_last_of("/\\"));

    // Construct the paths based on the cpp file's directory
    std::string csv_path = cppDirectory + "/../data/exploration_map.csv";
    std::string txt_path = cppDirectory + "/../../nbv_net/data/exploration_map.txt";

    visioncraft::saveExplorationMapToFiles(exploration_map, csv_path, txt_path);


    // Retrieve the surface shell map and calculate the coverage score
    auto surface_shell_map = model_loader.getSurfaceShellOctomap();
    double coverage_score = visioncraft::getCoverageScore(exploration_map, surface_shell_map);
    std::cout << "Coverage = " << coverage_score << std::endl;

    return 0;
}
