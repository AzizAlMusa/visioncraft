#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <memory>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include "visioncraft/model.h" // Include model loader for accessing meshes, point clouds, and octomaps
#include "visioncraft/viewpoint.h"    // Include viewpoint for managing viewpoint data
#include <vtkSmartPointer.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkAxesActor.h>
#include <vtkCamera.h>
#include <vtkFrustumSource.h>
#include <vtkPolyDataMapper.h>


namespace visioncraft {

/**
 * @class Visualizer
 * @brief A class for visualizing 3D data including viewpoints, point clouds, meshes, and octomaps.
 *
 * This class provides an interface for visualizing various 3D structures using VTK.
 * It integrates with Model and Viewpoint classes for seamless visualization.
 */
class Visualizer {
public:
    /**
     * @brief Constructor for Visualizer class.
     */
    Visualizer();

    /**
     * @brief Destructor for Visualizer class.
     */
    ~Visualizer();

    /**
     * @brief Set up the visualization window.
     * @param windowName The name of the visualization window (default is "3D Visualizer").
     */
    void initializeWindow(const std::string& windowName = "3D Visualizer");

    /**
     * @brief Add a viewpoint to the visualization.
     * @param viewpoint The viewpoint object containing position and orientation.
     * @param showFrustum Whether to show the frustum for the viewpoint (default is true).
     * @param showAxes Whether to show the axes for the viewpoint (default is true).
     */
    void addViewpoint(const visioncraft::Viewpoint& viewpoint, bool showFrustum = true, bool showAxes = true);

    /**
     * @brief Add multiple viewpoints to the visualization.
     * @param viewpoints A vector of viewpoint objects.
     */
    void addMultipleViewpoints(const std::vector<visioncraft::Viewpoint>& viewpoints);

    /**
     * @brief Visualize the rays generated from a viewpoint.
     * 
     * This function renders the rays as lines originating from the viewpoint's position
     * and extending to the end points of each ray.
     * 
     * @param viewpoint The viewpoint object containing the position and orientation from which the rays are generated.
     * @param color The RGB color of the rays as an Eigen::Vector3d (e.g., {1.0, 0.0, 0.0} for red).
     */
    void showRays(visioncraft::Viewpoint& viewpoint, const Eigen::Vector3d& color);

    /**
     * @brief Visualizes the ray endpoints as voxels along the ray paths generated from the viewpoint.
     * Each ray endpoint is visualized as a voxel (cube) in the 3D space.
     * 
     * @param viewpoint The viewpoint generating the rays.
     * @param voxel_size The size of each voxel (cube) representing the ray endpoints.
     * @param color The color of the voxels (RGB).
     */
    void showRayVoxels(visioncraft::Viewpoint& viewpoint, const std::shared_ptr<octomap::ColorOcTree>& octomap, const Eigen::Vector3d& color);


    /**
     * @brief Performs raycasting for the given viewpoints and visualizes the hit voxels in light green.
     * @param octomap A shared pointer to the octomap used for raycasting.
     */
    void showViewpointHits(const std::shared_ptr<octomap::ColorOcTree>& octomap);

   /**
     * @brief Add a mesh to the visualization from the Model, with an optional color.
     * @param model The model loader containing the mesh to be visualized.
     * @param color Optional color (RGB values) for the mesh (default: gray).
     */
    void addMesh(const visioncraft::Model& model, const Eigen::Vector3d& color = Eigen::Vector3d(0.8, 0.8, 0.8));

    /**
     * @brief Add a point cloud to the visualization from the Model, with an optional color.
     * @param model The model loader containing the point cloud to be visualized.
     * @param color Optional color (RGB values) for the point cloud (default: green).
     */
    void addPointCloud(const visioncraft::Model& model, const Eigen::Vector3d& color = Eigen::Vector3d(0.0, 1.0, 0.0));

    /**
     * @brief Add an octomap to the visualization from the Model, with an optional color.
     * @param model The model loader containing the octomap to be visualized.
     * @param color Optional color (RGB values) for the cubes in the octomap (default: node color).
     */
    void addOctomap(const visioncraft::Model& model, const Eigen::Vector3d& color = Eigen::Vector3d(-1, -1, -1));

    /**
     * @brief Show the VoxelGridGPU data on the visualizer.
     *
     * @param model The Model object containing the VoxelGridGPU.
     * @param color The color of the voxels for visualization.
     */
    void showGPUVoxelGrid(const visioncraft::Model& model, const Eigen::Vector3d& color);

    /**
     * @brief Visualize the MetaVoxelMap in the 3D space.
     * 
     * This function iterates through the MetaVoxelMap and visualizes each voxel as a cube.
     * Each voxel can be colored based on properties in the MetaVoxel or a default color.
     * 
     * @param model The model loader containing the MetaVoxelMap.
     * @param defaultColor Optional color (RGB values) for voxels without specific properties.
     */
    void addVoxelMap(const visioncraft::Model& model, const Eigen::Vector3d& defaultColor = Eigen::Vector3d(0.0, 0.0, 1.0));

    /**
     * @brief Visualize the MetaVoxelMap with color variation based on a specified property.
     * 
     * This function iterates through the MetaVoxelMap and visualizes each voxel as a cube,
     * adjusting the voxel color intensity based on the specified property value. The color 
     * darkens or lightens according to the property's value, either in a specified range 
     * or automatically normalized to the minimum and maximum property values found.
     * 
     * @param model The model loader containing the MetaVoxelMap.
     * @param property_name The name of the property in MetaVoxel to base the color intensity on.
     * @param baseColor The base RGB color to adjust for intensity; defaults to green.
     * @param minScale Optional minimum scale value for the property, defining the lightest color. 
     *                 If set to -1.0, the function calculates this from the property values.
     * @param maxScale Optional maximum scale value for the property, defining the darkest color. 
     *                 If set to -1.0, the function calculates this from the property values.
     */
    void addVoxelMapProperty(const visioncraft::Model& model, const std::string& property_name,
                                    const Eigen::Vector3d& baseColor = Eigen::Vector3d(0.0, 1.0, 0.0),
                                    const Eigen::Vector3d& propertyColor = Eigen::Vector3d(1.0, 1.0, 1.0), 
                                    float minScale = -1.0, float maxScale = -1.0);


    /**
     * @brief Set the background color of the visualization.
     * @param color The color to set as the background (RGB values).
     */
    void setBackgroundColor(const Eigen::Vector3d& color);

    /**
     * @brief Set the color of the viewpoint frustum.
     * @param color The color to set for the frustum (RGB values).
     */
    void setViewpointFrustumColor(const Eigen::Vector3d& color);

    void initializeRenderWindowInteractor();


    /**
     * @brief Start the rendering process for the visualization.
     */
    void render();

private:
    vtkSmartPointer<vtkRenderer> renderer; ///< The VTK renderer used for visualization.
    vtkSmartPointer<vtkRenderWindow> renderWindow; ///< The VTK render window.
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor; ///< The VTK render window interactor.
    /**
     * @brief Helper function to add a frustum for a viewpoint.
     * @param position The position of the viewpoint.
     * @param orientation The orientation matrix of the viewpoint.
     */
    void showFrustum(const visioncraft::Viewpoint& viewpoint);

    /**
     * @brief Helper function to add axes for a viewpoint.
     * @param position The position of the viewpoint.
     * @param orientation The orientation matrix of the viewpoint.
     */
    void showAxes(const Eigen::Vector3d& position, const Eigen::Matrix3d& orientation);

    vtkSmartPointer<vtkActor> octomapActor_;  ///< Store the actor for the octomap
    vtkSmartPointer<vtkUnsignedCharArray> octomapColors_;  ///< Store the color array for the octomap voxels
    vtkSmartPointer<vtkPolyData> octomapPolyData_;  ///< Store the poly data for easy access to voxel points

};

} // namespace visioncraft

#endif // VISUALIZER_H