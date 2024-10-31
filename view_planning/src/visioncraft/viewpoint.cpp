#include "visioncraft/viewpoint.h"
#include <cmath>
#include <omp.h>
#include <unordered_set>

#include <set>
#include <tuple>

#include <chrono>
#include <iostream>

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


// void launchGenerateRaysOnGPU(const float* position, const float* orientation, 
//                              double hfov, double vfov, int resolution_width, int resolution_height,
//                              double near_plane, double far_plane, float* rays, int total_pixels);

// void launchGenerateRaysOnGPU(const float* position, const float* forward, const float* right, const float* up, 
//                              float hfov, float vfov, int ds_width, int ds_height,
//                              float near_plane, float far_plane, float* rays, int total_pixels, 
//                              const VoxelGridGPU& voxelGridGPU, int3* host_hit_voxels);

void launchGenerateRaysOnGPU(const float* position, const float* forward, const float* right, const float* up,
                             float hfov, float vfov, int ds_width, int ds_height,
                             float near_plane, float far_plane,
                             const VoxelGridGPU& voxelGridGPU, int3* host_hit_voxels, unsigned int& host_hit_count);

std::unordered_map<octomap::OcTreeKey, bool, octomap::OcTreeKey::KeyHash> Viewpoint::performRaycastingOnGPU(const Model& model) {

    // Fetch the VoxelGridGPU from the model loader
    const VoxelGridGPU& voxelGridGPU = model.getGPUVoxelGrid();

    // Start preparing GPU data
    auto start_prepare = std::chrono::high_resolution_clock::now();
    Eigen::Vector3d position;
    Eigen::Matrix3d orientation;
    double hfov, vfov, near_plane, far_plane;
    std::pair<int, int> resolution;

    // Get the data from the viewpoint
    getGPUCompatibleData(position, orientation, hfov, vfov, resolution, near_plane, far_plane);

    Eigen::Vector3d viewpointDirection = orientation.col(2);  // Forward direction
    Eigen::Vector3d referenceVector = (std::abs(viewpointDirection.z()) == 1.0)
                                      ? Eigen::Vector3d(1, 0, 0)
                                      : Eigen::Vector3d(0, 0, 1);
    Eigen::Vector3d rightVector = viewpointDirection.cross(referenceVector).normalized();
    Eigen::Vector3d upVector = rightVector.cross(viewpointDirection).normalized();

    float h_position[3] = {static_cast<float>(position.x()), static_cast<float>(position.y()), static_cast<float>(position.z())};
    float h_forward[3] = {static_cast<float>(viewpointDirection.x()), static_cast<float>(viewpointDirection.y()), static_cast<float>(viewpointDirection.z())};
    float h_right[3] = {static_cast<float>(rightVector.x()), static_cast<float>(rightVector.y()), static_cast<float>(rightVector.z())};
    float h_up[3] = {static_cast<float>(upVector.x()), static_cast<float>(upVector.y()), static_cast<float>(upVector.z())};


    // Start downsample resolution process
    auto ds_resolution = getDownsampledResolution();
    int ds_width = ds_resolution.first;
    int ds_height = ds_resolution.second;
    int total_pixels = ds_width * ds_height;
    // std::cout << "Resolution: " << ds_width << " x " << ds_height << std::endl;
  

    // Allocate memory for hit voxels on the host
    int3* host_hit_voxels = new int3[total_pixels];
    unsigned int host_hit_count = 0;

    // Start raycasting on GPU
    launchGenerateRaysOnGPU(h_position, h_forward, h_right, h_up, static_cast<float>(hfov), static_cast<float>(vfov),
                            ds_width, ds_height, static_cast<float>(near_plane),
                            static_cast<float>(far_plane),
                            voxelGridGPU, host_hit_voxels, host_hit_count);


    // Start processing hit voxels

    std::set<std::tuple<int, int, int>> unique_hit_voxels;

    for (unsigned int i = 0; i < host_hit_count; ++i) {
        int x = host_hit_voxels[i].x;
        int y = host_hit_voxels[i].y;
        int z = host_hit_voxels[i].z;
        unique_hit_voxels.insert(std::make_tuple(x, y, z));
    }

    // std::cout << "Total unique hit voxels: " << unique_hit_voxels.size() << std::endl;

    // Free allocated memory
    delete[] host_hit_voxels;

    // End total function timer

    // Since we're not transferring rays back, we can return an empty vector or regenerate rays if needed
    std::vector<Eigen::Vector3d> rays; // Empty vector in this case

    GPU_hits_ = unique_hit_voxels; // Store the hit voxels in the member variable
    hits_ = model.convertGPUHitsToOctreeKeys(unique_hit_voxels);  // Convert hit voxels to OctoMap format

    return hits_;
}

/**
 * @brief Performs raycasting from the viewpoint to detect if the rays intersect with any occupied voxels in the octomap.
 * 
 * @param octomap A shared pointer to an octomap::ColorOcTree representing the 3D environment.
 * @param use_parallel If true, raycasting will be performed using multithreading for better performance.
 * @return A map where the key is the voxel key (octomap::OcTreeKey), and the value is a boolean indicating if the voxel was hit.
 */
std::unordered_map<octomap::OcTreeKey, bool, octomap::OcTreeKey::KeyHash> Viewpoint::performRaycasting(
    const Model& model, bool use_parallel) {
    
    // Get the octomap from the model loader
    std::shared_ptr<octomap::ColorOcTree> octomap = model.getSurfaceShellOctomap();
    
    // If rays have not been generated yet, generate them.
    if (rays_.empty()) {
        rays_ = generateRays(); // Generate rays based on the current viewpoint parameters.
    }

    // Clear the previous hit results
    hits_.clear();

    int hit_count = 0;

    // Check if multithreading is enabled
    if (use_parallel) {
        // Multithreading version: split the work into multiple threads

        // Determine the number of available hardware threads.
        const int num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;  // Vector to store the threads.
        std::vector<std::unordered_map<octomap::OcTreeKey, bool, octomap::OcTreeKey::KeyHash>> thread_results(num_threads);  // Results per thread.

        // Determine the number of rays each thread will process.
        int batch_size = rays_.size() / num_threads;

        // Split the rays into batches for each thread.
        for (int i = 0; i < num_threads; ++i) {
            // Define the start and end of the batch for this thread
            auto begin = rays_.begin() + i * batch_size;
            auto end = (i == num_threads - 1) ? rays_.end() : begin + batch_size;

            // Reserve space in the result vector for this thread
            thread_results[i].reserve(std::distance(begin, end));

            // Launch a thread to process the assigned batch of rays
            threads.emplace_back([&, begin, end, i]() {
                for (auto it = begin; it != end; ++it) {
                    // Set up the ray origin (viewpoint position) and the ray direction
                    octomap::point3d ray_origin(position_.x(), position_.y(), position_.z());
                    octomap::point3d ray_end(it->x(), it->y(), it->z());

                    // Variable to store the hit point
                    octomap::point3d hit;
                    double ray_length = (ray_end - ray_origin).norm();  // Calculate the length of the ray
                    
                    // Perform the raycasting
                    bool is_hit = octomap->castRay(ray_origin, ray_end - ray_origin, hit, true, ray_length);

                    // Reverse lookup of the key from the hit position in the Octomap
                    octomap::OcTreeKey key = octomap->coordToKey(hit);  // Use the hit point, not ray_end

                    // If the ray hits an occupied voxel, mark it as true
                    if (is_hit) {
                        thread_results[i][key] = true;

                        // Retrieve the hit voxel
                        octomap::ColorOcTreeNode* hitNode = octomap->search(hit);
                        if (hitNode) {
                   
                            // Optionally set color for visualization
                            hitNode->setColor(0, 255, 0);  // Set to green
                        }
                    } else {
                        // If it doesn't hit, mark the voxel as false (miss)
                        thread_results[i][key] = false;
                    }
                }
            });
        }

        // Wait for all threads to complete execution
        for (auto& thread : threads) {
            thread.join();
        }

        // Combine the results from all threads into the final hits_ map
        hits_.clear();
        for (const auto& thread_result : thread_results) {
            hits_.insert(thread_result.begin(), thread_result.end());
        }

    } else {
        // Sequential version: process the rays one by one without multithreading
        for (const auto& ray : rays_) {
            octomap::point3d ray_origin(position_.x(), position_.y(), position_.z());
            octomap::point3d ray_end(ray.x(), ray.y(), ray.z());

            // Variable to store the hit point
            octomap::point3d hit;
            bool is_hit = octomap->castRay(ray_origin, ray_end - ray_origin, hit, true, (ray_end - ray_origin).norm());

            // Reverse lookup of the key from the hit position in the Octomap
            octomap::OcTreeKey key = octomap->coordToKey(hit);  // Use the hit point, not ray_end

            // If the ray hits an occupied voxel, mark it as true
            if (is_hit) {
                hits_[key] = true;

                // Retrieve the hit voxel
                octomap::ColorOcTreeNode* hitNode = octomap->search(hit);
                if (hitNode) {
                
                    // Optionally set color for visualization
                    hitNode->setColor(0, 255, 0);  // Set to green
                }
            } else {
                // If it doesn't hit, mark the voxel as false (miss)
                hits_[key] = false;
            }
        }
    }

    // Return the map of hit results (both true and false)
    return hits_;
}





} // namespace visioncraft
