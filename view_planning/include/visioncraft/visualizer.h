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

#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkSphereSource.h>

#include <open3d/Open3D.h>

namespace visioncraft {


struct Vector3dHash {
    std::size_t operator()(const Eigen::Vector3d& vec) const {
        std::size_t h1 = std::hash<double>{}(vec.x());
        std::size_t h2 = std::hash<double>{}(vec.y());
        std::size_t h3 = std::hash<double>{}(vec.z());
        return h1 ^ (h2 << 1) ^ (h3 << 2); // Combine hashes
    }
};

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
     * @brief Process any pending interaction events.
     *
     * This function allows the interactor to process events during each render step,
     * enabling user control of the camera and window.
     */
    void processEvents();

    /**
     * @brief Add a viewpoint to the visualization.
     * @param viewpoint The viewpoint object containing position and orientation.
     * @param showFrustum Whether to show the frustum for the viewpoint (default is true).
     * @param showAxes Whether to show the axes for the viewpoint (default is true).
     */
    void addViewpoint(const visioncraft::Viewpoint& viewpoint, bool showFrustum = true, bool showAxes = true,  bool showPosition = false, bool showDirection = false);

    void updateViewpoint(const visioncraft::Viewpoint& viewpoint, bool updateFrustum = true, bool updateAxes = true , bool updatePosition = false, bool updateDirection = false);

    /**
     * @brief Add multiple viewpoints to the visualization.
     * @param viewpoints A vector of viewpoint objects.
     */
    void addMultipleViewpoints(const std::vector<visioncraft::Viewpoint>& viewpoints);


    /**
     * @brief Remove all added viewpoints from the visualization.
     */
    void removeViewpoints();

    void removeViewpoint(const visioncraft::Viewpoint& viewpoint);



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
     * @brief Visualize the rays generated from a viewpoint in parallel.
     * 
    */
    void showRaysParallel(visioncraft::Viewpoint& viewpoint);

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
     * @brief Remove the voxel map property from the visualization.
     */
    void removeVoxelMapProperty();


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
    
     /**
     * @brief TODO
     */
    void renderStep();


    /**
     * @brief Starts the asynchronous rendering loop in a separate thread.
     */
    void startAsyncRendering();

    /**
     * @brief Stops the asynchronous rendering thread safely.
     */
    void stopAsyncRendering();

    /**
     * @brief Get the render window interactor for managing interactive events.
     * @return A pointer to the vtkRenderWindowInteractor.
     */
    vtkSmartPointer<vtkRenderWindowInteractor> getRenderWindowInteractor() const {return renderWindowInteractor;}

     /**
     * @brief Getter for the renderer.
     * @return A pointer to the renderer.
     */
    vtkSmartPointer<vtkRenderer> getRenderer() const { return renderer; }

    /**
     * @brief Overlay text on the visualization window at a specified location.
     * 
     * This function adds a text overlay to the visualization window, allowing customization 
     * of the position, font size, and color. The text is displayed on top of the rendered scene.
     * 
     * @param text The text string to display.
     * @param x X-coordinate for text position in normalized [0.0, 1.0] viewport coordinates.
     * @param y Y-coordinate for text position in normalized [0.0, 1.0] viewport coordinates.
     * @param fontSize The font size of the text overlay.
     * @param color The color of the text in RGB format.
     */
    void addOverlayText(const std::string& text, double x, double y, int fontSize, const Eigen::Vector3d& color);

    /**
     * @brief Remove all text overlays from the visualization.
     *
     * This function removes all previously added text overlays to clear the visualization
     * or prepare for new overlays.
     */
    void removeOverlayTexts();

    /**
     * @brief Visualize the potential field on a sphere around the model.
     * 
    */
    void visualizePotentialOnSphere(const visioncraft::Model& model, float sphere_radius, const std::string& property_name, const std::unordered_map<octomap::OcTreeKey, Eigen::Vector3d, octomap::OcTreeKey::KeyHash>& voxelToSphereMap);
    
    void removePotentialSphere();

    void visualizeVoxelToSphereMapping(
    visioncraft::Model& model,
    const octomap::OcTreeKey& key,
    const std::unordered_map<octomap::OcTreeKey, Eigen::Vector3d, octomap::OcTreeKey::KeyHash>& voxelToSphereMap); 

    void removeVoxelToSphereMapping();

    void visualizeVoxelNormals(
    visioncraft::Model& model,
    double normalLength,
    const Eigen::Vector3d& color,
    const octomap::OcTreeKey& key);

    void showGeodesic(
    const visioncraft::Viewpoint& viewpoint,
    const Eigen::Vector3d& spherePoint,
    float sphereRadius);

    void removeGeodesic();

    void visualizePaths(const std::unordered_map<int, std::vector<Eigen::Vector3d>>& paths, float sphereRadius);

private:

    /**
     * @brief Helper function to add a frustum for a viewpoint.
     * @param position The position of the viewpoint.
     * @param orientation The orientation matrix of the viewpoint.
     */
    std::vector<vtkSmartPointer<vtkActor>> showFrustum(const visioncraft::Viewpoint& viewpoint);

    /**
     * @brief Helper function to add axes for a viewpoint.
     * @param position The position of the viewpoint.
     * @param orientation The orientation matrix of the viewpoint.
     */
    std::vector<vtkSmartPointer<vtkActor>> showAxes(const Eigen::Vector3d& position, const Eigen::Matrix3d& orientation);

    std::vector<vtkSmartPointer<vtkActor>> showSphere(const visioncraft::Viewpoint& viewpoint);
    std::vector<vtkSmartPointer<vtkActor>> showArrow(const visioncraft::Viewpoint& viewpoint);
    vtkSmartPointer<vtkRenderer> renderer; ///< The VTK renderer used for visualization.
    vtkSmartPointer<vtkRenderWindow> renderWindow; ///< The VTK render window.
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor; ///< The VTK render window interactor.
      
    vtkSmartPointer<vtkActor> octomapActor_;  ///< Store the actor for the octomap
    vtkSmartPointer<vtkUnsignedCharArray> octomapColors_;  ///< Store the color array for the octomap voxels
    vtkSmartPointer<vtkPolyData> octomapPolyData_;  ///< Store the poly data for easy access to voxel points


    std::vector<vtkSmartPointer<vtkActor>> viewpointActors_; ///< Store actors for all viewpoints
    vtkSmartPointer<vtkActor> voxelMapPropertyActor_ = nullptr; ///< Store actor for the voxel map property
    std::unordered_map<int, std::vector<vtkSmartPointer<vtkActor>>> viewpointActorMap_;


    std::thread renderThread_; ///< Thread for asynchronous rendering
    bool stopRendering_ = false; ///< Flag to control rendering loop termination

    std::vector<vtkSmartPointer<vtkTextActor>> overlayTextActors_; ///< Store text actors for overlay text.

    vtkSmartPointer<vtkActor> potentialSphereActor; // Class member to store the actor for removal
    vtkSmartPointer<vtkActor> projectedPointsActor; // Store the actor for removal later
    vtkSmartPointer<vtkActor> voxelToSphereMappingActor_;
    std::vector<vtkSmartPointer<vtkActor>> voxelToSphereMappingActors_;
    vtkSmartPointer<vtkActor> voxelNormalsActor_;
    vtkSmartPointer<vtkActor> geodesicActor_;
    vtkSmartPointer<vtkActor> pathActor_;

};

} // namespace visioncraft

#endif // VISUALIZER_H