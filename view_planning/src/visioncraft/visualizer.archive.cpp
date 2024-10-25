#include "visioncraft/visualizer.h"
#include <open3d/geometry/LineSet.h>
#include <open3d/geometry/Octree.h>

namespace visioncraft {

Visualizer::Visualizer() {
    initialize();
}

Visualizer::~Visualizer() {
    // Cleanup if necessary
}

void Visualizer::initialize() {
    visualizer_ = std::make_shared<open3d::visualization::Visualizer>();
    visualizer_->CreateVisualizerWindow("Visualizer", 1280, 720);
}



void Visualizer::addWorldFrame(double axis_length) {
    // Create a LineSet for the world axes representation
    auto axis = std::make_shared<open3d::geometry::LineSet>();
    

    // Define the origin and the endpoints of the axes
    Eigen::Vector3d origin = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d x_axis = Eigen::Vector3d(axis_length, 0, 0);  // X-axis (Red)
    Eigen::Vector3d y_axis = Eigen::Vector3d(0, axis_length, 0);  // Y-axis (Green)
    Eigen::Vector3d z_axis = Eigen::Vector3d(0, 0, axis_length);  // Z-axis (Blue)

    // Add the origin and axis points to the LineSet
    axis->points_.push_back(origin);
    axis->points_.push_back(x_axis);
    axis->points_.push_back(y_axis);
    axis->points_.push_back(z_axis);

    // Define the lines (each axis)
    axis->lines_.push_back(Eigen::Vector2i(0, 1));  // X-axis line
    axis->lines_.push_back(Eigen::Vector2i(0, 2));  // Y-axis line
    axis->lines_.push_back(Eigen::Vector2i(0, 3));  // Z-axis line

    // Set the colors for each axis
    axis->colors_.push_back(Eigen::Vector3d(1, 0, 0));  // Red for X-axis
    axis->colors_.push_back(Eigen::Vector3d(0, 1, 0));  // Green for Y-axis
    axis->colors_.push_back(Eigen::Vector3d(0, 0, 1));  // Blue for Z-axis

    // Add the world axes to the visualizer
    visualizer_->AddGeometry(axis);
}

void Visualizer::addLineSet(const std::shared_ptr<open3d::geometry::LineSet>& lineSet) {
    geometries_.push_back(lineSet);
}

void Visualizer::addMesh(const std::shared_ptr<open3d::geometry::TriangleMesh>& mesh) {
    geometries_.push_back(mesh);
}

void Visualizer::addPointCloud(const std::shared_ptr<open3d::geometry::PointCloud>& pointCloud) {
    geometries_.push_back(pointCloud);
}

void Visualizer::addVoxelGrid(const std::shared_ptr<open3d::geometry::VoxelGrid>& voxelGrid) {
    geometries_.push_back(voxelGrid);
}


void Visualizer::addOctomap(const std::shared_ptr<open3d::geometry::VoxelGrid>& voxelGrid) {

    //TODO: Remove this code
    
    // double voxel_resolution = voxelGrid->voxel_size_ * 2;
    // // Determine the bounding box size
    // Eigen::Vector3d min_bound = voxelGrid->GetMinBound();
    // Eigen::Vector3d max_bound = voxelGrid->GetMaxBound();
    // Eigen::Vector3d bounding_box_size = max_bound - min_bound;

    // // Calculate the maximum depth of the octree based on the voxel resolution
    // size_t max_depth = static_cast<size_t>(std::ceil(std::log2(bounding_box_size.maxCoeff() / voxel_resolution)));

    // std::cout << "Bounding Box Size: " << bounding_box_size.transpose() << std::endl;
    // std::cout << "Voxel Resolution: " << voxel_resolution << std::endl;
    // std::cout << "Calculated Max Depth: " << max_depth << std::endl;

    // // Create the octree with the calculated resolution
    // auto octree = std::make_shared<open3d::geometry::Octree>(max_depth, min_bound, bounding_box_size.maxCoeff());

    // // Use the CreateFromVoxelGrid function to populate the octree
    // octree->CreateFromVoxelGrid(*voxelGrid);

    // // Print the number of leaf nodes to verify the octree creation
    // size_t leaf_count = 0;
    // octree->Traverse([&leaf_count](const std::shared_ptr<open3d::geometry::OctreeNode>& node, const std::shared_ptr<open3d::geometry::OctreeNodeInfo>& node_info) {
    //     if (std::dynamic_pointer_cast<open3d::geometry::OctreeLeafNode>(node)) {
    //         ++leaf_count;
    //     }
    //     return false; // continue traversal
    // });

    // std::cout << "Total leaf nodes in the octree: " << leaf_count << std::endl;

    // // Add the octree to the geometries for visualization
    // geometries_.push_back(octree);
}




void Visualizer::addViewpoint(const Viewpoint& viewpoint) {
    viewpoints_.push_back(viewpoint);
}


void Visualizer::show(bool show_frustum) {
   
    // Add the world axes to the scene
    addWorldFrame(1.0);  // You can adjust the length of the world axes

     for (const auto& geometry : geometries_) {
        visualizer_->AddGeometry(geometry);
    }

    // Visualization logic for viewpoints
    for (const auto& viewpoint : viewpoints_) {
        // Create a LineSet for the 3D axis representation
        auto axis = std::make_shared<open3d::geometry::LineSet>();
        
        // Define the origin and the endpoints of the axes
        Eigen::Vector3d origin = viewpoint.getPosition();
        Eigen::Vector3d x_axis = origin + viewpoint.getOrientationMatrix().col(0) * 1.0; // X-axis (Red)
        Eigen::Vector3d y_axis = origin + viewpoint.getOrientationMatrix().col(1) * 1.0; // Y-axis (Green)
        Eigen::Vector3d z_axis = origin + viewpoint.getOrientationMatrix().col(2) * 1.0; // Z-axis (Blue)

        // Add the origin and axis points to the LineSet
        axis->points_.push_back(origin);
        axis->points_.push_back(x_axis);
        axis->points_.push_back(y_axis);
        axis->points_.push_back(z_axis);

        // Define the lines (each axis)
        axis->lines_.push_back(Eigen::Vector2i(0, 1)); // X-axis line
        axis->lines_.push_back(Eigen::Vector2i(0, 2)); // Y-axis line
        axis->lines_.push_back(Eigen::Vector2i(0, 3)); // Z-axis line

        // Set the colors for each axis
        axis->colors_.push_back(Eigen::Vector3d(1, 0, 0)); // Red for X-axis
        axis->colors_.push_back(Eigen::Vector3d(0, 1, 0)); // Green for Y-axis
        axis->colors_.push_back(Eigen::Vector3d(0, 0, 1)); // Blue for Z-axis

        // Add the axis to the visualizer
        visualizer_->AddGeometry(axis);

        // Add frustum only if show_frustum is true
        if (show_frustum) {
            auto frustum_corners = viewpoint.getFrustumCorners();
            auto lines = std::make_shared<open3d::geometry::LineSet>();
            lines->points_.resize(8);
            for (size_t i = 0; i < 8; ++i) {
                lines->points_[i] = frustum_corners[i];
            }

            lines->lines_ = {
                {0, 1}, {1, 2}, {2, 3}, {3, 0}, // Near plane
                {4, 5}, {5, 6}, {6, 7}, {7, 4}, // Far plane
                {0, 4}, {1, 5}, {2, 6}, {3, 7}  // Connections between planes
            };
            lines->PaintUniformColor(Eigen::Vector3d(0, 1, 0));
            visualizer_->AddGeometry(lines);
        }
    }



    visualizer_->Run();
    visualizer_->DestroyVisualizerWindow();
}


} // namespace visioncraft
