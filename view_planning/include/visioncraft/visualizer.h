#ifndef VISIONCRAFT_VISUALIZER_H
#define VISIONCRAFT_VISUALIZER_H

#include <memory>
#include <vector>
#include <open3d/Open3D.h>
#include "visioncraft/model_loader.h"
#include "visioncraft/viewpoint.h"

namespace visioncraft {

/**
 * @brief Class for visualizing 3D models, point clouds, voxel grids, and other representations.
 * 
 * This class provides methods to add and visualize different geometries using Open3D.
 */
class Visualizer {
public:
    /**
     * @brief Default constructor for Visualizer class.
     */
    Visualizer();

    /**
     * @brief Destructor for Visualizer class.
     */
    ~Visualizer();

    /**
     * @brief Add the world coordinate frame to the visualization.
     * 
     * @param axis_length The length of the axes to be visualized.
     */
    void addWorldFrame(double axis_length = 1.0);

    /**
     * @brief Add a line set to the visualization.
     * 
     * This function is used to visualize sets of lines, such as rays or edges, in the scene.
     * 
     * @param lineSet The line set to be added.
     */
    void addLineSet(const std::shared_ptr<open3d::geometry::LineSet>& lineSet);

    /**
     * @brief Add a mesh to the visualization.
     * 
     * @param mesh The mesh to be added.
     */
    void addMesh(const std::shared_ptr<open3d::geometry::TriangleMesh>& mesh);

    /**
     * @brief Add a point cloud to the visualization.
     * 
     * @param pointCloud The point cloud to be added.
     */
    void addPointCloud(const std::shared_ptr<open3d::geometry::PointCloud>& pointCloud);

    /**
     * @brief Add a voxel grid to the visualization.
     * 
     * @param voxelGrid The voxel grid to be added.
     */
    void addVoxelGrid(const std::shared_ptr<open3d::geometry::VoxelGrid>& voxelGrid);

    /**
     * @brief Add an OctoMap to the visualization.
     * 
     * @param octoMap The OctoMap to be added.
     */ 
    void addOctomap(const std::shared_ptr<open3d::geometry::VoxelGrid>& voxelGrid); //const std::shared_ptr<octomap::ColorOcTree>& octoMap

    /**
     * @brief Add a viewpoint to the visualization.
     * 
     * @param viewpoint The viewpoint to be added.
     */
    void addViewpoint(const Viewpoint& viewpoint);

    /**
     * @brief Render the visualization.
     */
    void show(bool show_frustum = false);

private:
    // Initialize the visualization window and other states.
    void initialize();

    std::shared_ptr<open3d::visualization::Visualizer> visualizer_; ///< Open3D visualizer instance.
    std::vector<std::shared_ptr<open3d::geometry::Geometry>> geometries_; ///< List of geometries to be visualized.
    std::vector<Viewpoint> viewpoints_; ///< List of viewpoints to be visualized.
};

} // namespace visioncraft

#endif // VISIONCRAFT_VISUALIZER_H
