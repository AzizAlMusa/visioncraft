#include <iostream>
#include <fstream>
#include "visioncraft/model_loader.h"
#include "visioncraft/viewpoint.h"
#include <octomap/OcTreeKey.h>
#include <octomap/octomap_types.h>

namespace visioncraft {

class ExplorationMapManager {
public:
    ExplorationMapManager(const std::string& model_file, const std::string& nbv_list_file) 
        : model_file_(model_file), nbv_list_file_(nbv_list_file) {}

    bool initialize() {
        // Load the 3D model, generate a point cloud, and create the exploration map
        if (!model_loader_.loadExplorationModel(model_file_, 20000, 32)) {
            std::cerr << "Failed to load mesh." << std::endl;
            return false;
        }

        exploration_map_ = model_loader_.getExplorationMap();
        octomap_ = model_loader_.getOctomap();

        // Save scaling factor
        double scale_factor = 0.025 / exploration_map_->getResolution();
        saveScalingFactor(scale_factor, "../data/scaling_factor.txt");

        // Read viewpoints from file
        viewpoints_ = readViewpointsFromFile(nbv_list_file_);

        // Set downsample factor for each viewpoint
        for (auto& viewpoint : viewpoints_) {
            viewpoint.setDownsampleFactor(4.0);
        }

        return true;
    }

    void performRaycasting() {
        for (auto& viewpoint : viewpoints_) {
            performGuidedRaycasting(exploration_map_, octomap_, viewpoint);
        }
    }

    void saveExplorationData() const {
        std::string csv_path = "../data/exploration_map.csv";
        std::string txt_path = "../../nbv_net/data/exploration_map.txt";

        saveExplorationMapToFiles(exploration_map_, csv_path, txt_path);
    }

    void calculateCoverage() const {
        auto surface_shell_map = model_loader_.getSurfaceShellOctomap();
        double coverage_score = getCoverageScore(exploration_map_, surface_shell_map);
        std::cout << "Coverage = " << coverage_score << std::endl;
    }


    std::vector<Viewpoint> readViewpointsFromFile(const std::string& filename) {
        std::vector<Viewpoint> viewpoints;
        std::ifstream file(filename);

        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return viewpoints;
        }

        double x, y, z, lookAt_x, lookAt_y, lookAt_z;
        while (file >> x >> y >> z >> lookAt_x >> lookAt_y >> lookAt_z) {
            Eigen::Vector3d position(x, y, z);
            Eigen::Vector3d lookAt(lookAt_x, lookAt_y, lookAt_z);
            viewpoints.emplace_back(position, lookAt);
        }

        file.close();
        return viewpoints;
    }

    void saveScalingFactor(double scaling_factor, const std::string& file_path) const {
        std::ofstream file(file_path);
        if (file.is_open()) {
            file << scaling_factor;
            file.close();
            std::cout << "Scaling factor saved to " << file_path << ": " << scaling_factor << std::endl;
        } else {
            std::cerr << "Error opening file to save scaling factor: " << file_path << std::endl;
        }
    }

    void performGuidedRaycasting(std::shared_ptr<octomap::ColorOcTree>& exploration_map,
                                 std::shared_ptr<octomap::ColorOcTree>& octomap,
                                 Viewpoint& viewpoint) const {
        int hit_count = 0;
        int free_voxels_updated = 0;

        auto rays = viewpoint.generateRays();
        std::cout << "Number of rays generated: " << rays.size() << std::endl;

        double min_x, min_y, min_z, max_x, max_y, max_z;
        exploration_map->getMetricMin(min_x, min_y, min_z);
        exploration_map->getMetricMax(max_x, max_y, max_z);

        for (const auto& ray_end : rays) {
            octomap::point3d ray_origin(viewpoint.getPosition().x(), viewpoint.getPosition().y(), viewpoint.getPosition().z());
            octomap::point3d ray_target(ray_end.x(), ray_end.y(), ray_end.z());
            octomap::KeyRay keyRay;

            if (octomap->computeRayKeys(ray_origin, ray_target, keyRay)) {
                for (const auto& key : keyRay) {
                    octomap::point3d voxel_center = octomap->keyToCoord(key);

                    if (voxel_center.x() >= min_x && voxel_center.x() <= max_x &&
                        voxel_center.y() >= min_y && voxel_center.y() <= max_y &&
                        voxel_center.z() >= min_z && voxel_center.z() <= max_z) {

                        auto node = exploration_map->search(key);
                        auto octomap_node = octomap->search(key);

                        if (node && octomap_node) {
                            node->setLogOdds(1.0);  
                            hit_count++;
                            break;
                        } else if (node) {
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
    }

    void saveExplorationMapToFiles(const std::shared_ptr<octomap::ColorOcTree>& exploration_map,
                                   const std::string& csv_path,
                                   const std::string& txt_path) const {
        std::ofstream csv_file(csv_path);
        std::ofstream txt_file(txt_path);

        csv_file << "x,y,z,value\n";
        double exploration_map_resolution = exploration_map->getResolution();
        double scale_factor = 0.025 / exploration_map_resolution;

        int leaf_count = 0;
        for (auto it = exploration_map->begin_leafs(); it != exploration_map->end_leafs(); ++it) {
            if (it.getDepth() == exploration_map->getTreeDepth()) {
                octomap::point3d coord = it.getCoordinate();
                double value;

                if (it->getOccupancy() == 0.5) {
                    value = 0.5;
                } else if (exploration_map->isNodeOccupied(*it)) {
                    value = 1.0;
                } else {
                    value = 0.0;
                }

                csv_file << coord.x() << "," << coord.y() << "," << coord.z() << "," << value << "\n";

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
        if (leaf_count == 32768) {
            std::cout << "CSV writing complete: Correct number of leaf nodes written." << std::endl;
        } else {
            std::cerr << "CSV writing complete: Incorrect number of leaf nodes written." << std::endl;
        }
    }

    double getCoverageScore(std::shared_ptr<octomap::ColorOcTree>& exploration_map,
                            std::shared_ptr<octomap::ColorOcTree>& surface_map) const {
        int matched_occupied_voxels = 0;
        int total_occupied_voxels_surface = 0;

        for (auto it = surface_map->begin_leafs(); it != surface_map->end_leafs(); ++it) {
            if (surface_map->isNodeOccupied(*it)) {
                total_occupied_voxels_surface++;
                octomap::point3d surface_voxel_center = it.getCoordinate();
                
                auto exploration_node = exploration_map->search(surface_voxel_center);
                if (exploration_node && exploration_node->getOccupancy() > 0.5) {
                    matched_occupied_voxels++;
                }
            }
        }

        return static_cast<double>(matched_occupied_voxels) / total_occupied_voxels_surface;
    }

private:
    ModelLoader model_loader_;
    std::shared_ptr<octomap::ColorOcTree> exploration_map_;
    std::shared_ptr<octomap::ColorOcTree> octomap_;
    std::vector<Viewpoint> viewpoints_;

    std::string model_file_;
    std::string nbv_list_file_;
};

}  // namespace visioncraft

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model_file> <nbv_list_file>" << std::endl;
        return 1;
    }

    visioncraft::ExplorationMapManager map_manager(argv[1], argv[2]);

    if (!map_manager.initialize()) {
        return -1;
    }

    map_manager.performRaycasting();
    map_manager.saveExplorationData();
    map_manager.calculateCoverage();

    return 0;
}
