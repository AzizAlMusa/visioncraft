#include "visioncraft/viewpoint.h"
#include <cmath>
#include <omp.h>

namespace visioncraft {

Viewpoint::Viewpoint() 
    : position_(Eigen::Vector3d::Zero()), 
      orientation_matrix_(Eigen::Matrix3d::Identity()),
      quaternion_(Eigen::Quaterniond::Identity()),
      near_(350.0), 
      far_(900.0), 
      resolution_width_(2448), 
      resolution_height_(2048),
      downsample_factor_(32.0),
      hfov_(44.8), 
      vfov_(42.6) {}

Viewpoint::Viewpoint(const Eigen::Vector3d& position, const Eigen::Matrix3d& orientation, 
                     double near, double far, 
                     int resolution_width, int resolution_height,
                     double hfov, double vfov)
    : position_(position), 
      orientation_matrix_(orientation), 
      quaternion_(Eigen::Quaterniond(orientation)),
      near_(near), 
      far_(far), 
      resolution_width_(resolution_width), 
      resolution_height_(resolution_height),
      downsample_factor_(32.0),
      hfov_(hfov), 
      vfov_(vfov) {}

Viewpoint::Viewpoint(const Eigen::VectorXd& position_yaw_pitch_roll,
                     double near, double far, 
                     int resolution_width, int resolution_height,
                     double hfov, double vfov)
    : position_(position_yaw_pitch_roll.head<3>()), 
      near_(near), 
      far_(far), 
      resolution_width_(resolution_width), 
      resolution_height_(resolution_height),
      downsample_factor_(32.0),
      hfov_(hfov), 
      vfov_(vfov) {

    // Extract yaw, pitch, roll from the vector
    double yaw = position_yaw_pitch_roll[3];
    double pitch = position_yaw_pitch_roll[4];
    double roll = position_yaw_pitch_roll[5];

    // Convert yaw, pitch, roll to a rotation matrix
    orientation_matrix_ = 
        Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()) *
        Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());

    // Initialize quaternion from the rotation matrix
    quaternion_ = Eigen::Quaterniond(orientation_matrix_);
}


Viewpoint::Viewpoint(const Eigen::Vector3d& position, const Eigen::Vector3d& lookAt, 
                     const Eigen::Vector3d& up, double near, double far, 
                     int resolution_width, int resolution_height,
                     double hfov, double vfov)
    : position_(position), 
      near_(near), 
      far_(far), 
      resolution_width_(resolution_width), 
      resolution_height_(resolution_height),
      downsample_factor_(32.0),
      hfov_(hfov), 
      vfov_(vfov) {
    setLookAt(lookAt, up);
}

Viewpoint::~Viewpoint() {}

void Viewpoint::setPosition(const Eigen::Vector3d& position) {
    position_ = position;
}

Eigen::Vector3d Viewpoint::getPosition() const {
    return position_;
}

void Viewpoint::setOrientation(const Eigen::Matrix3d& orientation) {
    orientation_matrix_ = orientation;
    quaternion_ = Eigen::Quaterniond(orientation);
}

void Viewpoint::setOrientation(const Eigen::Quaterniond& quaternion) {
    quaternion_ = quaternion;
    orientation_matrix_ = quaternion.toRotationMatrix();
}

void Viewpoint::setLookAt(const Eigen::Vector3d& lookAt, const Eigen::Vector3d& up) {
    Eigen::Vector3d forward = (lookAt - position_).normalized();

    // Check if the forward vector is parallel to the Z-axis
    if (std::abs(forward.dot(Eigen::Vector3d(0, 0, 1))) > 0.9999) {
        // If forward is parallel to the Z-axis, choose a different up vector
        Eigen::Vector3d adjustedUp = Eigen::Vector3d(0, 1, 0); // Use Y-axis as the up vector
        Eigen::Vector3d right = adjustedUp.cross(forward).normalized();
        adjustedUp = forward.cross(right).normalized();

        orientation_matrix_.col(0) = right;
        orientation_matrix_.col(1) = adjustedUp;
        orientation_matrix_.col(2) = forward;
        
    } else {
        Eigen::Vector3d right = up.cross(forward).normalized();
        Eigen::Vector3d adjustedUp = forward.cross(right).normalized();

        orientation_matrix_.col(0) = right;
        orientation_matrix_.col(1) = adjustedUp;
        orientation_matrix_.col(2) = forward;
    }
    
    quaternion_ = Eigen::Quaterniond(orientation_matrix_);
}

Eigen::Matrix3d Viewpoint::getOrientationMatrix() const {
    return orientation_matrix_;
}

Eigen::Quaterniond Viewpoint::getOrientationQuaternion() const {
    return quaternion_;
}

Eigen::Matrix4d Viewpoint::getTransformationMatrix() const {
    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    transformation.block<3,3>(0,0) = orientation_matrix_;
    transformation.block<3,1>(0,3) = position_;
    return transformation;
}

Eigen::Vector3d Viewpoint::getOrientationEuler() const {
    return orientation_matrix_.eulerAngles(0, 1, 2);
}

void Viewpoint::setNearPlane(double near) {
    near_ = near;
}

double Viewpoint::getNearPlane() const {
    return near_;
}

void Viewpoint::setFarPlane(double far) {
    far_ = far;
}

double Viewpoint::getFarPlane() const {
    return far_;
}

void Viewpoint::setResolution(int width, int height) {
    resolution_width_ = width;
    resolution_height_ = height;
}

std::pair<int, int> Viewpoint::getResolution() const {
    return {resolution_width_, resolution_height_};
}

void Viewpoint::setDownsampleFactor(double factor) {
    downsample_factor_ = factor;
}

double Viewpoint::getDownsampleFactor() const {
    return downsample_factor_;
}

std::pair<int, int> Viewpoint::getDownsampledResolution() const {
    return {static_cast<int>(resolution_width_ / downsample_factor_), 
            static_cast<int>(resolution_height_ / downsample_factor_)};
}

void Viewpoint::setHorizontalFieldOfView(double hfov) {
    hfov_ = hfov;
}

double Viewpoint::getHorizontalFieldOfView() const {
    return hfov_;
}

void Viewpoint::setVerticalFieldOfView(double vfov) {
    vfov_ = vfov;
}

double Viewpoint::getVerticalFieldOfView() const {
    return vfov_;
}

std::vector<Eigen::Vector3d> Viewpoint::getFrustumCorners() const {
    std::vector<Eigen::Vector3d> corners;

    double near_height = 2 * near_ * tan(vfov_ * M_PI / 360.0);
    double near_width = 2 * near_ * tan(hfov_ * M_PI / 360.0);

    double far_height = 2 * far_ * tan(vfov_ * M_PI / 360.0);
    double far_width = 2 * far_ * tan(hfov_ * M_PI / 360.0);

    corners.push_back(position_ + orientation_matrix_ * Eigen::Vector3d(-near_width / 2, -near_height / 2, near_));
    corners.push_back(position_ + orientation_matrix_ * Eigen::Vector3d(near_width / 2, -near_height / 2, near_));
    corners.push_back(position_ + orientation_matrix_ * Eigen::Vector3d(near_width / 2, near_height / 2, near_));
    corners.push_back(position_ + orientation_matrix_ * Eigen::Vector3d(-near_width / 2, near_height / 2, near_));

    corners.push_back(position_ + orientation_matrix_ * Eigen::Vector3d(-far_width / 2, -far_height / 2, far_));
    corners.push_back(position_ + orientation_matrix_ * Eigen::Vector3d(far_width / 2, -far_height / 2, far_));
    corners.push_back(position_ + orientation_matrix_ * Eigen::Vector3d(far_width / 2, far_height / 2, far_));
    corners.push_back(position_ + orientation_matrix_ * Eigen::Vector3d(-far_width / 2, far_height / 2, far_));

    return corners;
}


std::vector<Eigen::Vector3d> Viewpoint::generateRays() {
    std::vector<Eigen::Vector3d> rays;

    // Use downsampled resolution
    int ds_width = static_cast<int>(resolution_width_ / downsample_factor_);
    int ds_height = static_cast<int>(resolution_height_ / downsample_factor_);

    rays.reserve(ds_width * ds_height);

    // Calculate the direction vector from the camera to the lookAt point
    Eigen::Vector3d viewpointDirection = orientation_matrix_.col(2); // Forward direction

    // Handle the edge case where the viewpointDirection is parallel to (0, 0, 1)
    Eigen::Vector3d referenceVector = (std::abs(viewpointDirection.z()) == 1.0) ? Eigen::Vector3d(1, 0, 0) : Eigen::Vector3d(0, 0, 1);

    // Calculate the right and up vectors
    Eigen::Vector3d rightVector = viewpointDirection.cross(referenceVector).normalized();
    Eigen::Vector3d upVector = rightVector.cross(viewpointDirection).normalized();

    // Calculate the step size for the horizontal and vertical angles
    double hStep = hfov_ * M_PI / 180.0 / static_cast<double>(ds_width);
    double vStep = vfov_ * M_PI / 180.0 / static_cast<double>(ds_height);

    // Generate rays for each pixel
    for (int j = 0; j < ds_height; ++j) {
        for (int i = 0; i < ds_width; ++i) {
            double hAngle = -0.5 * hfov_ * M_PI / 180.0 + i * hStep;
            double vAngle = -0.5 * vfov_ * M_PI / 180.0 + j * vStep;

            // Compute the ray direction in world space
            Eigen::Vector3d rayDir = viewpointDirection + rightVector * std::tan(hAngle) + upVector * std::tan(vAngle);
            rayDir.normalize();

            // Scale the ray to intersect with the far plane
            double scale = far_ / viewpointDirection.dot(rayDir);  // Adjust the scaling factor for each ray
            Eigen::Vector3d rayEnd = position_ + rayDir * scale;

            // Store the ray end point
            rays.push_back(rayEnd);
        }
    }

    // Store the generated rays in the rays_ member variable
    rays_ = rays;

    
    return rays;
}





/**
 * @brief Performs raycasting from the viewpoint to detect if the rays intersect with any occupied voxels in the octomap.
 * 
 * @param octomap A shared pointer to an octomap::ColorOcTree representing the 3D environment.
 * @param use_parallel If true, raycasting will be performed using multithreading for better performance.
 * @return A vector of pairs where the first element is a boolean indicating if the ray hit an occupied voxel,
 *         and the second element is the 3D coordinates of the hit point (or a zero vector if no hit).
 */
std::vector<std::pair<bool, Eigen::Vector3d>> Viewpoint::performRaycasting(const std::shared_ptr<octomap::ColorOcTree>& octomap, bool use_parallel) {
    
    // If rays have not been generated yet, generate them.
    if (rays_.empty()) {
        rays_ = generateRays(); // Generate rays based on the current viewpoint parameters.
    }

    // Prepare a vector to store the results of the raycasting.
    // Each element in the vector will be a pair, where the first element is a boolean (hit or miss),
    // and the second element is the 3D coordinates of the hit point (or a zero vector if no hit).
    std::vector<std::pair<bool, Eigen::Vector3d>> hit_results(rays_.size());

    // If multithreading is enabled
    if (use_parallel) {
        // Determine the number of available hardware threads.
        const int num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads; // Vector to store the threads.
        std::vector<std::vector<std::pair<bool, Eigen::Vector3d>>> thread_results(num_threads); // Vector to store results from each thread.

        // Determine the number of rays each thread will process.
        int batch_size = rays_.size() / num_threads;

        // Split the rays into batches for each thread.
        for (int i = 0; i < num_threads; ++i) {
            auto begin = rays_.begin() + i * batch_size;
            auto end = (i == num_threads - 1) ? rays_.end() : begin + batch_size;

            // Reserve space in the result vector for each thread.
            thread_results[i].reserve(std::distance(begin, end));

            // Launch a thread to process the assigned batch of rays.
            threads.emplace_back([&, begin, end, i]() {
                for (auto it = begin; it != end; ++it) {
                    // Set up the ray origin (viewpoint position) and the ray direction.
                    octomap::point3d ray_origin(position_.x(), position_.y(), position_.z());
                    octomap::point3d ray_end(it->x(), it->y(), it->z());

                    // Variable to store the hit point.
                    octomap::point3d hit;
                    double ray_length = (ray_end - ray_origin).norm(); // Calculate the length of the ray.
                    
                    // Perform the raycasting.
                    bool is_hit = octomap->castRay(ray_origin, ray_end - ray_origin, hit, true, ray_length);

                    // If the ray hits an occupied voxel, store the hit point. Otherwise, store a zero vector.
                    if (is_hit) {
                        thread_results[i].emplace_back(true, Eigen::Vector3d(hit.x(), hit.y(), hit.z()));
                    } else {
                        thread_results[i].emplace_back(false, Eigen::Vector3d::Zero());
                    }
                }
            });
        }

        // Wait for all threads to complete execution.
        for (auto& thread : threads) {
            thread.join();
        }

        // Combine the results from all threads into the final hit_results vector.
        hit_results.clear();
        for (const auto& result : thread_results) {
            hit_results.insert(hit_results.end(), result.begin(), result.end());
        }

    } else {
        // If multithreading is not enabled, perform raycasting sequentially.
        for (int i = 0; i < rays_.size(); ++i) {
            const auto& ray = rays_[i];
            octomap::point3d ray_origin(position_.x(), position_.y(), position_.z());
            octomap::point3d ray_end(ray.x(), ray.y(), ray.z());

            // Variable to store the hit point.
            octomap::point3d hit;
            // Perform the raycasting.
            bool is_hit = octomap->castRay(ray_origin, ray_end - ray_origin, hit, true, (ray_end - ray_origin).norm());

            // If the ray hits an occupied voxel, store the hit point. Otherwise, store a zero vector.
            if (is_hit) {
                hit_results[i] = std::make_pair(true, Eigen::Vector3d(hit.x(), hit.y(), hit.z()));
            } else {
                hit_results[i] = std::make_pair(false, Eigen::Vector3d::Zero());
            }
        }
    }

    // Return the vector containing the results of the raycasting.
    return hit_results;
}




} // namespace visioncraft
