#ifndef VISIONCRAFT_VIEWPOINT_H
#define VISIONCRAFT_VIEWPOINT_H

#include <Eigen/Dense>
#include <memory>
#include <open3d/Open3D.h>
#include <octomap/ColorOcTree.h>
#include <model.h>
#include <visibility_manager.h>
#include <cuda_runtime.h>


namespace visioncraft {

/**
 * @brief Class representing a viewpoint for depth camera positioning and orientation.
 * 
 * This class provides functionality to define and manipulate the position and orientation
 * of a depth camera viewpoint in a 3D space. It also includes camera properties like near
 * and far planes, resolution, and downsampling for simulation purposes.
 */
class Viewpoint : public std::enable_shared_from_this<Viewpoint>   {
public:
    /**
     * @brief Default constructor for Viewpoint class.
     * 
     * Initializes the viewpoint with default position, orientation, and camera properties.
     */
    Viewpoint();

    /**
     * @brief Constructor with position, orientation, and camera properties.
     * 
     * @param position The position of the viewpoint as an Eigen::Vector3d.
     * @param orientation The orientation of the viewpoint as a rotation matrix (Eigen::Matrix3d).
     * @param near The near plane distance of the camera frustum in millimeters (default: 350.0).
     * @param far The far plane distance of the camera frustum in millimeters (default: 900.0).
     * @param resolution_width The width resolution of the camera (default: 2448).
     * @param resolution_height The height resolution of the camera (default: 2048).
     * @param hfov The horizontal field of view of the camera in degrees (default: 44.8).
     * @param vfov The vertical field of view of the camera in degrees (default: 42.6).
     */
    Viewpoint(const Eigen::Vector3d& position, const Eigen::Matrix3d& orientation, 
              double near = 350.0, double far = 900.0, 
              int resolution_width = 2448, int resolution_height = 2048,
              double hfov = 44.8, double vfov = 42.6);
   
    /**
     * @brief Constructor with combined position and Euler angles (yaw, pitch, roll).
     * 
     * This constructor initializes the viewpoint using a single Eigen::Vector3d 
     * where the first three elements represent the position (x, y, z), and the last 
     * three elements represent the Euler angles (yaw, pitch, roll).
     * 
     * @param position_yaw_pitch_roll A combined Eigen::Vector3d where:
     *        - position_yaw_pitch_roll[0] = x position
     *        - position_yaw_pitch_roll[1] = y position
     *        - position_yaw_pitch_roll[2] = z position
     *        - position_yaw_pitch_roll[3] = yaw (rotation around Z axis) in radians
     *        - position_yaw_pitch_roll[4] = pitch (rotation around Y axis) in radians
     *        - position_yaw_pitch_roll[5] = roll (rotation around X axis) in radians
     * @param near The near plane distance of the camera frustum in millimeters (default: 350.0).
     * @param far The far plane distance of the camera frustum in millimeters (default: 900.0).
     * @param resolution_width The width resolution of the camera (default: 2448).
     * @param resolution_height The height resolution of the camera (default: 2048).
     * @param hfov The horizontal field of view of the camera in degrees (default: 44.8).
     * @param vfov The vertical field of view of the camera in degrees (default: 42.6).
     */
    Viewpoint(const Eigen::VectorXd& position_yaw_pitch_roll,
            double near = 350.0, double far = 900.0, 
            int resolution_width = 2448, int resolution_height = 2048,
            double hfov = 44.8, double vfov = 42.6);

    /**
     * @brief Constructor with position and lookAt point.
     * 
     * @param position The position of the viewpoint as an Eigen::Vector3d.
     * @param lookAt The target point the viewpoint is oriented towards.
     * @param up The up direction for the viewpoint (default: Eigen::Vector3d::UnitY()).
     * @param near The near plane distance of the camera frustum in millimeters (default: 350.0).
     * @param far The far plane distance of the camera frustum in millimeters (default: 900.0).
     * @param resolution_width The width resolution of the camera (default: 2448).
     * @param resolution_height The height resolution of the camera (default: 2048).
     * @param hfov The horizontal field of view of the camera in degrees (default: 44.8).
     * @param vfov The vertical field of view of the camera in degrees (default: 42.6).
     */
    Viewpoint(const Eigen::Vector3d& position, const Eigen::Vector3d& lookAt, 
              const Eigen::Vector3d& up = -Eigen::Vector3d::UnitZ(),
              double near = 350.0, double far = 900.0, 
              int resolution_width = 2448, int resolution_height = 2048,
              double hfov = 44.8, double vfov = 42.6);

    /**
     * @brief Destructor for Viewpoint class.
     */
    ~Viewpoint();

    // ID Methods
    int getId() const  {return id_;}; // Get the unique ID of the viewpoint

    // Position and Orientation Methods

    /**
     * @brief Set the position of the viewpoint.
     * 
     * @param position The position of the viewpoint as an Eigen::Vector3d.
     */
    void setPosition(const Eigen::Vector3d& position);

    /**
     * @brief Get the position of the viewpoint.
     * 
     * @return The position of the viewpoint as an Eigen::Vector3d.
     */
    Eigen::Vector3d getPosition() const;

    /**
     * @brief Set the orientation of the viewpoint using a rotation matrix.
     * 
     * @param orientation The orientation of the viewpoint as a rotation matrix (Eigen::Matrix3d).
     */
    void setOrientation(const Eigen::Matrix3d& orientation);

    /**
     * @brief Set the orientation of the viewpoint using quaternions.
     * 
     * @param quaternion The orientation of the viewpoint as an Eigen::Quaterniond.
     */
    void setOrientation(const Eigen::Quaterniond& quaternion);

    /**
     * @brief Set the orientation of the viewpoint using a lookAt target point.
     * 
     * @param lookAt The target point the viewpoint is oriented towards.
     * @param up The up direction for the viewpoint (default: Eigen::Vector3d::UnitY()).
     */
    void setLookAt(const Eigen::Vector3d& lookAt, const Eigen::Vector3d& up = -Eigen::Vector3d::UnitZ());

    /**
     * @brief Get the orientation of the viewpoint as a rotation matrix.
     * 
     * @return The orientation of the viewpoint as a rotation matrix (Eigen::Matrix3d).
     */
    Eigen::Matrix3d getOrientationMatrix() const;

    /**
     * @brief Get the orientation of the viewpoint as quaternions.
     * 
     * @return The orientation of the viewpoint as an Eigen::Quaterniond.
     */
    Eigen::Quaterniond getOrientationQuaternion() const;

    /**
     * @brief Compute the transformation matrix for the viewpoint.
     * 
     * @return The transformation matrix as an Eigen::Matrix4d.
     */
    Eigen::Matrix4d getTransformationMatrix() const;

    /**
     * @brief Convert the orientation to Euler angles.
     * 
     * @return The orientation of the viewpoint as Euler angles (Eigen::Vector3d).
     */
    Eigen::Vector3d getOrientationEuler() const;

    // Camera Properties Methods

    /**
     * @brief Set the near plane distance of the camera frustum.
     * 
     * @param near The near plane distance in millimeters.
     */
    void setNearPlane(double near);

    /**
     * @brief Get the near plane distance of the camera frustum.
     * 
     * @return The near plane distance in millimeters.
     */
    double getNearPlane() const;

    /**
     * @brief Set the far plane distance of the camera frustum.
     * 
     * @param far The far plane distance in millimeters.
     */
    void setFarPlane(double far);

    /**
     * @brief Get the far plane distance of the camera frustum.
     * 
     * @return The far plane distance in millimeters.
     */
    double getFarPlane() const;

    /**
     * @brief Set the resolution of the camera.
     * 
     * @param width The width resolution of the camera.
     * @param height The height resolution of the camera.
     */
    void setResolution(int width, int height);

    /**
     * @brief Get the resolution of the camera.
     * 
     * @return A pair representing the width and height resolution of the camera.
     */
    std::pair<int, int> getResolution() const;

    /**
     * @brief Set the downsample factor for simulation.
     * 
     * @param factor The factor by which to downsample the resolution.
     */
    void setDownsampleFactor(double factor);

    /**
     * @brief Get the downsample factor for simulation.
     * 
     * @return The downsample factor.
     */
    double getDownsampleFactor() const;

    /**
     * @brief Get the downsampled resolution of the camera.
     * 
     * @return A pair representing the downsampled width and height resolution of the camera.
     */
    std::pair<int, int> getDownsampledResolution() const;

    /**
     * @brief Set the horizontal field of view (hFOV) of the camera.
     * 
     * @param hfov The horizontal field of view in degrees.
     */
    void setHorizontalFieldOfView(double hfov);

    /**
     * @brief Get the horizontal field of view (hFOV) of the camera.
     * 
     * @return The horizontal field of view in degrees.
     */
    double getHorizontalFieldOfView() const;

    /**
     * @brief Set the vertical field of view (vFOV) of the camera.
     * 
     * @param vfov The vertical field of view in degrees.
     */
    void setVerticalFieldOfView(double vfov);

    /**
     * @brief Get the vertical field of view (vFOV) of the camera.
     * 
     * @return The vertical field of view in degrees.
     */
    double getVerticalFieldOfView() const;

    /**
     * @brief Compute and return the frustum corners of the camera.
     * 
     * @return A vector of Eigen::Vector3d representing the frustum corners.
     */
    std::vector<Eigen::Vector3d> getFrustumCorners() const;


    /**
     * @brief Get the generated rays for the viewpoint.
     * 
     * @return A vector of Eigen::Vector3d representing the rays' end points from the viewpoint.
     */
    std::vector<Eigen::Vector3d> getRays() const {return rays_;}

    /**
     * @brief Perform raycasting with the generated rays on the provided octomap.
     * 
     * This method checks whether the rays generated from the viewpoint hit any occupied voxel in the octomap.
     * 
     * @param octomap The octomap to perform raycasting on.
     * @return A vector of booleans indicating whether each ray hit an occupied voxel (true) or not (false).
     */
    std::unordered_map<octomap::OcTreeKey, bool, octomap::OcTreeKey::KeyHash>  performRaycasting(const Model& model, bool parallelize = false) ;


    /**
     * @brief Get the hit results of the raycasting operation.
     * 
     * @return A vector of pairs indicating whether each ray hit an occupied voxel (true) or not (false) and the hit point.
     */
    std::unordered_map<octomap::OcTreeKey, bool, octomap::OcTreeKey::KeyHash> getHitResults() const {return hits_;}

    /**
     * @brief Generate rays for each pixel in the image plane.
     * 
     * This function calculates the direction of rays for each pixel in the image plane
     * based on the camera's intrinsic and extrinsic parameters.
     * 
     * @return A vector of Eigen::Vector3d representing the direction of rays from the camera's position.
     */
    std::vector<Eigen::Vector3d> generateRays();

     /**
     * @brief Get the generated rays as an octomap.
     * 
     * @return A octomap::ColorOcTree representing the rays.
     */
    std::shared_ptr<octomap::ColorOcTree> getRaysOctomap() const {return rays_octomap_;}


    /**
     * @brief Provide necessary data to the GPU for ray generation.
     * This method returns the minimal set of data needed for ray generation on the GPU.
     * 
     * @param position Output: the viewpoint position.
     * @param orientation Output: the orientation matrix of the viewpoint.
     * @param hfov Output: the horizontal field of view.
     * @param vfov Output: the vertical field of view.
     * @param resolution Output: a pair containing the resolution width and height.
     * @param near Output: the near plane distance.
     * @param far Output: the far plane distance.
     */
    void getGPUCompatibleData(Eigen::Vector3d& position, Eigen::Matrix3d& orientation, 
                              double& hfov, double& vfov, std::pair<int, int>& resolution,
                              double& near, double& far) const {
        position = position_;
        orientation = orientation_matrix_;
        hfov = hfov_;
        vfov = vfov_;
        resolution = {resolution_width_, resolution_height_};
        near = near_;
        far = far_;
    }


    /**
     * @brief Perform raycasting using GPU.
     * 
     * @return A vector of Eigen::Vector3d containing the ray endpoints generated on the GPU.
     */
    std::unordered_map<octomap::OcTreeKey, bool, octomap::OcTreeKey::KeyHash> performRaycastingOnGPU(const Model& model);


    /**
     * @brief Get the hit results of the raycasting operation performed on the GPU.
     * 
     * @return TODO
     */
    std::set<std::tuple<int, int, int>> getGPUHitResults() const {return GPU_hits_;}

    /**
     * @brief Add an observer to receive notifications about changes in raycasting hits.
     * 
     * Adds the provided observer to the list of observers that will be notified whenever
     * there is an update in the `hits_` variable, indicating visibility changes.
     * 
     * @param observer Shared pointer to the observer (VisibilityManager) to be added.
     */
    void addObserver(const std::shared_ptr<VisibilityManager>& observer);

    /**
     * @brief Remove an observer from the notification list.
     * 
     * Removes the specified observer from the list of observers, ensuring it no
     * longer receives updates when `hits_` changes.
     * 
     * @param observer Shared pointer to the observer (VisibilityManager) to be removed.
     */
    void removeObserver(const std::shared_ptr<VisibilityManager>& observer);

    /**
     * @brief Notify all registered observers of changes in the viewpoint’s visibility.
     * 
     * Iterates over all observers and calls their `updateVisibility` method to inform
     * them of the latest `hits_` results.
     */
    void notifyObservers();


private:

    static std::atomic<int> globalIdCounter_; ///< Static counter for unique IDs (thread-safe).
    int id_; ///< Unique ID for this viewpoint instance.


    // Viewpoint pose
    Eigen::Vector3d position_; ///< The position of the viewpoint.
    Eigen::Matrix3d orientation_matrix_; ///< The orientation of the viewpoint as a rotation matrix.
    Eigen::Quaterniond quaternion_; ///< The orientation of the viewpoint as a quaternion.

    // Camera properties
    double near_; ///< The near plane distance of the camera frustum in millimeters.
    double far_; ///< The far plane distance of the camera frustum in millimeters.
    int resolution_width_; ///< The width resolution of the camera.
    int resolution_height_; ///< The height resolution of the camera.
    double downsample_factor_; ///< The factor by which to downsample the resolution for simulation.
    double hfov_; ///< The horizontal field of view of the camera in degrees.
    double vfov_; ///< The vertical field of view of the camera in degrees.

    // Rays
    std::vector<Eigen::Vector3d> rays_; ///< Stores the generated rays for the viewpoint.
    std::unordered_map<octomap::OcTreeKey, bool, octomap::OcTreeKey::KeyHash> hits_; ///< Stores the hit results of the raycasting operation.
    std::set<std::tuple<int, int, int>> GPU_hits_; ///< Stores the hit results of the raycasting operation performed on the GPU.
    std::shared_ptr<octomap::ColorOcTree> rays_octomap_; ///< Stores the rays as an octomap (typically for visualization).


    // Observer
    std::unordered_set<std::shared_ptr<VisibilityManager>> observers_; ///< Set of registered observers.

};

} // namespace visioncraft





#endif // VISIONCRAFT_VIEWPOINT_H