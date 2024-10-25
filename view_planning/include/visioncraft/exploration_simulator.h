#ifndef EXPLORATION_SIMULATOR_H
#define EXPLORATION_SIMULATOR_H

#include <memory>
#include <vector>
#include "visioncraft/model_loader.h"
#include "visioncraft/viewpoint.h"
#include <octomap/ColorOcTree.h>

namespace visioncraft {

class ExplorationSimulator {
public:
    // Constructor
    ExplorationSimulator();

    // Load the 3D model and generate the exploration map
    bool loadModel(const std::string& model_file, int num_points = 20000, int grid_resolution = 32);

    // Set viewpoints
    void setViewpoints(const std::vector<Viewpoint>& viewpoints);

    // Perform raycasting and update the exploration map
    void performRaycasting();

    // Get exploration map data
    std::vector<std::tuple<double, double, double, double>> getExplorationMapData();

    // Calculate coverage score
    double getCoverageScore();

    // Get scaling factor
    double getScalingFactor() const;

private:
    // Member variables
    std::shared_ptr<octomap::ColorOcTree> exploration_map_;
    std::shared_ptr<octomap::ColorOcTree> octomap_;
    std::shared_ptr<octomap::ColorOcTree> surface_shell_map_;
    std::vector<Viewpoint> viewpoints_;
    double scaling_factor_;

    // Helper methods
    void performGuidedRaycasting(Viewpoint& viewpoint);
};

} // namespace visioncraft

#endif // EXPLORATION_SIMULATOR_H
