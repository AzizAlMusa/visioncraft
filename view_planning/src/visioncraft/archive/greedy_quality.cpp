#include "visioncraft/model.h"
#include "visioncraft/viewpoint.h"
#include "visioncraft/visibility_manager.h"
#include "visioncraft/visualizer.h"
#include <Eigen/Dense>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>

// Function to generate random positions around a sphere
std::vector<Eigen::Vector3d> generateRandomPositions(double radius, int count) {
    std::vector<Eigen::Vector3d> positions;
    for (int i = 0; i < count; ++i) {
        double theta = ((double) rand() / RAND_MAX) * 2 * M_PI;
        double phi = ((double) rand() / RAND_MAX) * M_PI;
        double x = radius * std::sin(phi) * std::cos(theta);
        double y = radius * std::sin(phi) * std::sin(theta);
        double z = radius * std::cos(phi);
        positions.emplace_back(x, y, z);
    }
    return positions;
}

// Function to safely get alignment property with a default fallback
double getVoxelAlignment(const visioncraft::Model& model, const octomap::OcTreeKey& voxelKey) {
    try {
        return boost::get<double>(model.getVoxelProperty(voxelKey, "alignment"));
    } catch (const boost::bad_get&) {
        // Return default value if alignment property is missing
        return 0.0;
    }
}

int main() {
    srand(time(nullptr)); // Initialize random seed

    // Initialize the visualizer
    visioncraft::Visualizer visualizer;
    visualizer.setBackgroundColor(Eigen::Vector3d(0.0, 0.0, 0.0));

    // Load the model
    visioncraft::Model model;
    std::cout << "Loading model..." << std::endl;
    model.loadModel("../models/gorilla.ply", 100000);
    std::cout << "Model loaded successfully." << std::endl;

    model.addVoxelProperty("alignment", 0.0);  // Initialize alignment property for all voxels

    // Initialize visibility manager
    auto visibilityManager = std::make_shared<visioncraft::VisibilityManager>(model);
    std::cout << "VisibilityManager initialized." << std::endl;

    // Threshold for good-quality alignment
    const double alignment_threshold = 0.95;
    const double quality_coverage_target = 0.95;  // Stop when 99.5% of voxels meet quality requirement

    std::vector<std::shared_ptr<visioncraft::Viewpoint>> selectedViewpoints;

    // Create a binary map for voxel quality status: true if the voxel meets quality, false otherwise
    std::unordered_map<octomap::OcTreeKey, bool, octomap::OcTreeKey::KeyHash> qualityMap;
    const size_t totalVoxelCount = model.getVoxelMap().size();
    size_t goodQualityVoxelCount = 0;

    int iteration = 0;

    while (static_cast<double>(goodQualityVoxelCount) / totalVoxelCount < quality_coverage_target) {
        std::cout << "\nIteration " << iteration + 1 << " - Searching for best viewpoint." << std::endl;

        auto randomPositions = generateRandomPositions(400.0, 10);
        std::shared_ptr<visioncraft::Viewpoint> bestViewpoint = nullptr;
        size_t bestNovelGoodQualityCount = 0;

        for (const auto& position : randomPositions) {
            auto viewpoint = std::make_shared<visioncraft::Viewpoint>(position, Eigen::Vector3d(0.0, 0.0, 0.0));
            viewpoint->setDownsampleFactor(4.0);

            visibilityManager->trackViewpoint(viewpoint);
            viewpoint->performRaycastingOnGPU(model);
            const auto& visibleVoxels = viewpoint->getHitResults();

            Eigen::Matrix3d orientation_matrix = viewpoint->getOrientationMatrix();
            Eigen::Vector3d viewpoint_z_axis = orientation_matrix.col(2);

            // Count novel good-quality voxels for this viewpoint
            size_t novelGoodQualityCount = 0;
            for (const auto& voxelKey : visibleVoxels) {
              
                if (qualityMap[voxelKey.first]) continue;  // Skip already covered good-quality voxels

                // Compute alignment score
                Eigen::Vector3d voxel_normal = boost::get<Eigen::Vector3d>(model.getVoxelProperty(voxelKey.first, "normal"));
                double alignment_score = std::abs(viewpoint_z_axis.dot(voxel_normal));

                // Safely set the alignment value
                model.setVoxelProperty(voxelKey.first, "alignment", alignment_score);

                // Count as novel good-quality voxel if it meets the alignment threshold
                if (alignment_score >= alignment_threshold) {
                    novelGoodQualityCount++;
                }
            }

            visibilityManager->untrackViewpoint(viewpoint);

            // Update best viewpoint if this viewpoint has more novel good-quality voxels
            if (novelGoodQualityCount > bestNovelGoodQualityCount) {
                bestNovelGoodQualityCount = novelGoodQualityCount;
                bestViewpoint = viewpoint;
            }
        }

        // Admit the best viewpoint and update quality map
        if (bestViewpoint) {
            visibilityManager->trackViewpoint(bestViewpoint);
            bestViewpoint->performRaycastingOnGPU(model);
            selectedViewpoints.push_back(bestViewpoint);

            // Update quality map with the newly covered good-quality voxels
            const auto& bestVisibleVoxels = bestViewpoint->getHitResults();
            Eigen::Matrix3d orientation_matrix = bestViewpoint->getOrientationMatrix();
            Eigen::Vector3d viewpoint_z_axis = orientation_matrix.col(2);

            size_t novelAddedCount = 0;
            for (const auto& voxelKey : bestVisibleVoxels) {
                if (qualityMap[voxelKey.first]) continue;  // Skip already covered voxels

                Eigen::Vector3d voxel_normal = boost::get<Eigen::Vector3d>(model.getVoxelProperty(voxelKey.first, "normal"));
                double alignment_score = std::abs(viewpoint_z_axis.dot(voxel_normal));

                if (alignment_score >= alignment_threshold) {
                    qualityMap[voxelKey.first] = true;
                    novelAddedCount++;
                }
            }

            // Update good-quality voxel count and coverage score
            goodQualityVoxelCount += novelAddedCount;
            double qualityCoverage = static_cast<double>(goodQualityVoxelCount) / totalVoxelCount;

            std::cout << "Quality coverage after iteration " << iteration + 1 << ": " << qualityCoverage << " / " << quality_coverage_target << std::endl;
        } else {
            std::cout << "No suitable viewpoint found in this iteration." << std::endl;
        }

        iteration++;
    }

    // Print final results
    std::cout << "\nFinal Quality Coverage Score: " << static_cast<double>(goodQualityVoxelCount) / totalVoxelCount << std::endl;
    std::cout << "Total Viewpoints Selected: " << selectedViewpoints.size() << std::endl;

    // Render the selected viewpoints and visible voxels
    Eigen::Vector3d baseColor(1.0, 1.0, 1.0);
    Eigen::Vector3d propertyColor(0.0, 0.0, 1.0);
    // visualizer.addVoxelMapProperty(model, "visibility", baseColor, propertyColor);
   

    visualizer.addVoxelMapProperty(model, "alignment",  Eigen::Vector3d(1.0, 0.0, 0.0) , Eigen::Vector3d(0.0, 1.0, 0.0), 0.0, 1.0);

    // for (const auto& viewpoint : selectedViewpoints) {
    //     visualizer.addViewpoint(*viewpoint, true, true);
    // }
    // visualizer.showGPUVoxelGrid(model, Eigen::Vector3d(0.0, 0.0, 1.0));
    visualizer.render();

    return 0;
}
