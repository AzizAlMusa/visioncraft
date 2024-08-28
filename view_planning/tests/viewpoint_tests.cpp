#include "visioncraft/model_loader.h"
#include "visioncraft/viewpoint.h"
#include <open3d/Open3D.h>
#include <iostream>

// Function to create axes visualization as LineSet
std::shared_ptr<open3d::geometry::LineSet> CreateAxesVisualization(const Eigen::Vector3d& position, const Eigen::Matrix3d& orientation, double axis_length) {
    auto line_set = std::make_shared<open3d::geometry::LineSet>();

    Eigen::Vector3d origin = position;
    Eigen::Vector3d x_axis = position + axis_length * orientation.col(0);
    Eigen::Vector3d y_axis = position + axis_length * orientation.col(1);
    Eigen::Vector3d z_axis = position + axis_length * orientation.col(2);

    line_set->points_.push_back(origin);
    line_set->points_.push_back(x_axis);
    line_set->points_.push_back(y_axis);
    line_set->points_.push_back(z_axis);

    line_set->lines_.push_back(Eigen::Vector2i(0, 1));
    line_set->lines_.push_back(Eigen::Vector2i(0, 2));
    line_set->lines_.push_back(Eigen::Vector2i(0, 3));

    line_set->colors_.push_back(Eigen::Vector3d(1, 0, 0));  // X-axis in red
    line_set->colors_.push_back(Eigen::Vector3d(0, 1, 0));  // Y-axis in green
    line_set->colors_.push_back(Eigen::Vector3d(0, 0, 1));  // Z-axis in blue

    return line_set;
}

// Function to create world axes visualization
std::shared_ptr<open3d::geometry::LineSet> CreateWorldAxesVisualization(double axis_length) {
    auto line_set = std::make_shared<open3d::geometry::LineSet>();

    Eigen::Vector3d origin = Eigen::Vector3d::Zero();
    Eigen::Vector3d x_axis = axis_length * Eigen::Vector3d(1, 0, 0);
    Eigen::Vector3d y_axis = axis_length * Eigen::Vector3d(0, 1, 0);
    Eigen::Vector3d z_axis = axis_length * Eigen::Vector3d(0, 0, 1);

    line_set->points_.push_back(origin);
    line_set->points_.push_back(x_axis);
    line_set->points_.push_back(y_axis);
    line_set->points_.push_back(z_axis);

    line_set->lines_.push_back(Eigen::Vector2i(0, 1));
    line_set->lines_.push_back(Eigen::Vector2i(0, 2));
    line_set->lines_.push_back(Eigen::Vector2i(0, 3));

    line_set->colors_.push_back(Eigen::Vector3d(1, 0, 0));  // X-axis in red
    line_set->colors_.push_back(Eigen::Vector3d(0, 1, 0));  // Y-axis in green
    line_set->colors_.push_back(Eigen::Vector3d(0, 0, 1));  // Z-axis in blue

    return line_set;
}

// Function to create frustum visualization
std::shared_ptr<open3d::geometry::LineSet> CreateFrustumVisualization(const visioncraft::Viewpoint& viewpoint) {
    auto line_set = std::make_shared<open3d::geometry::LineSet>();
    auto corners = viewpoint.getFrustumCorners();

    // Define the lines that make up the frustum
    std::vector<Eigen::Vector2i> lines = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},  // Near plane
        {4, 5}, {5, 6}, {6, 7}, {7, 4},  // Far plane
        {0, 4}, {1, 5}, {2, 6}, {3, 7}   // Connecting lines
    };

    for (const auto& corner : corners) {
        line_set->points_.push_back(corner);
        std::cout << "Frustum corner: " << corner.transpose() << std::endl;  // Debug print
    }

    for (const auto& line : lines) {
        line_set->lines_.push_back(line);
        std::cout << "Frustum line: " << line.transpose() << std::endl;  // Debug print
    }

    // Set the color of the lines to blue
    for (size_t i = 0; i < line_set->lines_.size(); ++i) {
        line_set->colors_.push_back(Eigen::Vector3d(0, 0, 1));  // Blue color
    }

    return line_set;
}

int main(int argc, char** argv) {

    // Define position and orientation using LookAt
    Eigen::Vector3d position(1.0, 0, 0);
    Eigen::Vector3d lookAt(0.0, 0.0, 0.0); // Define the point to look at
    Eigen::Vector3d up(0.0, 0.0, -1);   // Define the up direction as negative Z of the world frame to point down

    // Create a Viewpoint instance with default camera properties
    visioncraft::Viewpoint viewpoint(position, lookAt, up);

    // Set position and lookAt orientation
    viewpoint.setPosition(position);
    viewpoint.setLookAt(lookAt, up);

    // Set near and far planes
    viewpoint.setNearPlane(300.0);
    viewpoint.setFarPlane(1000.0);

    // Set resolution
    viewpoint.setResolution(1920, 1080);

    // Set downsample factor for simulation
    viewpoint.setDownsampleFactor(2.0);

    // Set horizontal and vertical FOV
    viewpoint.setHorizontalFieldOfView(44.8);
    viewpoint.setVerticalFieldOfView(42.6);

    // Get transformation matrix
    Eigen::Matrix4d transformation = viewpoint.getTransformationMatrix();

    // Output the transformation matrix
    std::cout << "Transformation Matrix:\n" << transformation << std::endl;

    // Output camera properties
    std::cout << "Near Plane: " << viewpoint.getNearPlane() << " mm\n";
    std::cout << "Far Plane: " << viewpoint.getFarPlane() << " mm\n";
    std::cout << "Resolution: " << viewpoint.getResolution().first << " x " << viewpoint.getResolution().second << "\n";
    std::cout << "Downsampled Resolution: " << viewpoint.getDownsampledResolution().first << " x " << viewpoint.getDownsampledResolution().second << "\n";
    std::cout << "Horizontal FOV: " << viewpoint.getHorizontalFieldOfView() << " degrees\n";
    std::cout << "Vertical FOV: " << viewpoint.getVerticalFieldOfView() << " degrees\n";

    // Get and output frustum corners
    std::vector<Eigen::Vector3d> frustumCorners = viewpoint.getFrustumCorners();
    std::cout << "Frustum Corners:\n";
    for (const auto& corner : frustumCorners) {
        std::cout << corner.transpose() << std::endl;
    }

    // Print the quaternion
    Eigen::Quaterniond quaternion = viewpoint.getOrientationQuaternion();
    std::cout << "Viewpoint Quaternion: "
              << quaternion.w() << " "
              << quaternion.x() << " "
              << quaternion.y() << " "
              << quaternion.z() << std::endl;

    // Create frustum visualization
    auto frustum = CreateFrustumVisualization(viewpoint);

    // Create axes visualization at the principal point
    auto axes = CreateAxesVisualization(viewpoint.getPosition(), viewpoint.getOrientationMatrix(), 0.25);

    // Create world axes visualization
    auto world_axes = CreateWorldAxesVisualization(0.25);

    // Combine frustum, axes, and world axes for visualization
    std::vector<std::shared_ptr<const open3d::geometry::Geometry>> geometries;
    // geometries.push_back(frustum);
    geometries.push_back(axes);
    geometries.push_back(world_axes);

    // Visualize the frustum and axes
    open3d::visualization::DrawGeometries(geometries, "Frustum and Axes Visualization");

    return 0;
}
