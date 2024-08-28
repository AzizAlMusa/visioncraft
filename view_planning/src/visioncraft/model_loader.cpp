#include "visioncraft/model_loader.h" // Include the ModelLoader header file

#include <iostream>
#include <iomanip>
#include <string>
#include <Eigen/Core>



namespace visioncraft {

// Constructor for the ModelLoader class
ModelLoader::ModelLoader() {
    std::cout << "ModelLoader constructor called." << std::endl;
}

// Destructor for the ModelLoader class
ModelLoader::~ModelLoader() {
    std::cout << "ModelLoader destructor called." << std::endl;
}

// Load a mesh from a file
bool ModelLoader::loadMesh(const std::string& file_path) {
    // Open3D supports multiple formats including STL and PLY
    auto mesh = open3d::io::CreateMeshFromFile(file_path);
    if (mesh == nullptr || mesh->vertices_.empty()) {
        std::cerr << "Error: Failed to load mesh data from file." << std::endl;
        return false;
    } else {
        meshData_ = std::make_shared<open3d::geometry::TriangleMesh>(*mesh);
        std::cout << "Mesh loaded successfully with " << meshData_->vertices_.size() << " vertices." << std::endl;
    
        auto aabb = meshData_->GetAxisAlignedBoundingBox();
        std::cout << "Bounding box: min = " << aabb.min_bound_.transpose() << ", max = " << aabb.max_bound_.transpose() << ", center = " << aabb.GetCenter().transpose() << ", extent = " << aabb.GetExtent().transpose() << std::endl;

         // Assign the bounding box information to the member variables
        minBound_ = octomap::point3d(aabb.min_bound_.x(), aabb.min_bound_.y(), aabb.min_bound_.z());
        maxBound_ = octomap::point3d(aabb.max_bound_.x(), aabb.max_bound_.y(), aabb.max_bound_.z());
        center_ = octomap::point3d(aabb.GetCenter().x(), aabb.GetCenter().y(), aabb.GetCenter().z());

        return true;
    }
}

// Load a 3D model and generate all necessary structures
bool ModelLoader::loadModel(const std::string& file_path, int num_samples, double resolution) {
    if (!loadMesh(file_path)) {
        return false;
    }
    return generateAllStructures(num_samples, resolution);
}

// Load a 3D model and generate all necessary structures for exploration
bool ModelLoader::loadExplorationModel(const std::string& file_path, int num_samples, int num_cells_per_side ){
    if (!loadMesh(file_path)) {
        return false;
    }
    return generateExplorationStructures(num_samples, num_cells_per_side);
}

// Generate all necessary structures from the loaded mesh
bool ModelLoader::generateAllStructures(int num_samples, double resolution) {
    bool success = true;
   
    success &= initializeRaycastingScene();
    success &= generatePointCloud(num_samples); // Number of samples for the point cloud

    if (resolution <= 0) {
        resolution = getAverageSpacing();
        std::cout << "Using default resolution: " << resolution << std::endl;
    }

    success &= generateVolumetricPointCloud();
    success &= generateVoxelGrid(resolution);
    success &= generateOctoMap(resolution);
    success &= generateVolumetricOctoMap(resolution);
    success &= generateSurfaceShellOctomap();
    success &= generateExplorationMap(resolution, getMinBound(), getMaxBound());
    return success;
}

bool ModelLoader::generateExplorationStructures(int num_samples, int num_cells_per_side) {
    bool success = true;

    success &= initializeRaycastingScene();
    success &= generatePointCloud(num_samples); // Number of samples for the point cloud
    success &= generateVolumetricPointCloud();
    success &= generateExplorationMap(num_cells_per_side, getMinBound(), getMaxBound());

    double resolution = getExplorationMapResolution();
    success &= generateOctoMap(resolution);
    success &= generateVoxelGrid(resolution);
    success &= generateVolumetricOctoMap(resolution);
    success &= generateSurfaceShellOctomap();
    
    return success;
}

// Initialize the raycasting scene with the current mesh
bool ModelLoader::initializeRaycastingScene() {
    auto mesh = getMeshData();
    if (!mesh) {
        std::cerr << "Error: Mesh data is not loaded." << std::endl;
        return false;
    }

    // Convert legacy mesh to tensor-based mesh
    auto tensor_mesh = open3d::t::geometry::TriangleMesh::FromLegacy(*mesh);

    // Initialize RaycastingScene
    raycasting_scene_ = std::make_shared<open3d::t::geometry::RaycastingScene>();

    // Add triangle mesh to RaycastingScene
    raycasting_scene_->AddTriangles(tensor_mesh);

    return true;
}

// Generate a point cloud using Poisson disk sampling
bool ModelLoader::generatePointCloud(int numSamples) {
    // Check if mesh data is loaded
    if (!meshData_) {
        std::cerr << "Error: Mesh data is not loaded." << std::endl;
        return false;
    }

    // Print some mesh data before running the Poisson sampler
    std::cout << "Mesh Data Information:" << std::endl;
    std::cout << "Number of Vertices: " << meshData_->vertices_.size() << std::endl;
    std::cout << "Number of Triangles: " << meshData_->triangles_.size() << std::endl;


  
    // Generate point cloud using Poisson disk sampling from Open3D
    std::cout << "Generating point cloud using Poisson disk sampling with " << numSamples << " samples..."  << std::endl;
    std::shared_ptr<open3d::geometry::PointCloud> pcd = meshData_->SamplePointsPoissonDisk(numSamples, 5.0, nullptr, true);

    // Check if the point cloud is generated successfully
    if (!pcd->HasPoints()) {
        std::cerr << "Failed to generate point cloud from mesh." << std::endl;
        return false;
    }


    // Get the average point spacing
    std::vector<double> nearestNeighborDistances = pcd->ComputeNearestNeighborDistance();
    double averageSpacing = std::accumulate(nearestNeighborDistances.begin(), nearestNeighborDistances.end(), 0.0) / nearestNeighborDistances.size();
    pointCloudSpacing_ = averageSpacing;


    // Estimate normals for the point cloud
    std::cout << "Generating normals..."  << std::endl ;
    pcd->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(averageSpacing * 5, 30));  // Adjust these parameters as needed

    // Correct oppositely oriented normals using signed distance
    std::cout << "Correcting misaligned normals..."  << std::endl ;
    pcd = correctNormalsUsingSignedDistance(pcd, averageSpacing * 0.01);


    // Store the generated point cloud
    pointCloud_ = pcd;

    std::cout << "Point cloud created successfully."  << std::endl ;
    
    return true;
}




std::shared_ptr<open3d::geometry::PointCloud> ModelLoader::correctNormalsUsingSignedDistance(std::shared_ptr<open3d::geometry::PointCloud> pointcloud, double epsilon) {
    if (!raycasting_scene_) {
        std::cerr << "Error: Raycasting scene is not initialized." << std::endl;
        return pointcloud;
    }

    // Preparing point and normals data from pointCloud
    auto points = pointcloud->points_;
    auto normals = pointcloud->normals_;

    // Convert std::vector<Eigen::Vector3d> to Open3D Tensor directly
    open3d::core::Device device("CPU:0");
    open3d::core::Tensor points_tensor = open3d::core::eigen_converter::EigenVector3dVectorToTensor(pointcloud->points_, open3d::core::Dtype::Float32, device);
    open3d::core::Tensor normals_tensor = open3d::core::eigen_converter::EigenVector3dVectorToTensor(pointcloud->normals_, open3d::core::Dtype::Float32, device);

    // Create offset points in the direction of the normals and opposite
    open3d::core::Tensor offset_points = points_tensor + epsilon * normals_tensor;
    open3d::core::Tensor offset_points_neg = points_tensor - epsilon * normals_tensor;

    // Compute signed distances for offset points
    auto sdf_pos = raycasting_scene_->ComputeSignedDistance(offset_points);
    auto sdf_neg = raycasting_scene_->ComputeSignedDistance(offset_points_neg);

    // Correct normals based on signed distances
    auto sdf_pos_data = sdf_pos.ToFlatVector<float>();
    auto sdf_neg_data = sdf_neg.ToFlatVector<float>();

    for (size_t i = 0; i < pointcloud->points_.size(); ++i) {
        if (sdf_pos_data[i] > 0 && sdf_neg_data[i] < 0) {
            continue; // Correct orientation
        } else {
            // Flip the normal
            pointcloud->normals_[i] *= -1;
        }
    }

    return pointcloud;
}

bool ModelLoader::generateVolumetricPointCloud() {
    // Ensure point cloud data is available
    if (!pointCloud_) {
        std::cerr << "Error: Point cloud data is not available." << std::endl;
        return false;
    }

    // Ensure raycasting scene is initialized
    if (!raycasting_scene_) {
        std::cerr << "Error: Raycasting scene is not initialized." << std::endl;
        return false;
    }

    // Get the average spacing from the existing point cloud
    double spacing = getAverageSpacing();

    // Get the bounding box of the mesh
    octomap::point3d min_bound = getMinBound();
    octomap::point3d max_bound = getMaxBound();

    // Create a new point cloud
    volumetricPointCloud_ = std::make_shared<open3d::geometry::PointCloud>();

    // Prepare a vector to store the points to be evaluated
    std::vector<float> points;

    // Iterate through each point in the bounding box
    for (double x = min_bound.x(); x <= max_bound.x(); x += spacing) {
        for (double y = min_bound.y(); y <= max_bound.y(); y += spacing) {
            for (double z = min_bound.z(); z <= max_bound.z(); z += spacing) {
                points.push_back(static_cast<float>(x));
                points.push_back(static_cast<float>(y));
                points.push_back(static_cast<float>(z));
            }
        }
    }

    // Convert the points to a tensor
    open3d::core::Tensor points_tensor(points, {static_cast<int64_t>(points.size() / 3), 3}, open3d::core::Dtype::Float32, open3d::core::Device("CPU:0"));

    // Compute the signed distances for all points
    open3d::core::Tensor sdf_values = raycasting_scene_->ComputeSignedDistance(points_tensor);

    // Filter points with negative signed distance values
    auto sdf_values_data = sdf_values.ToFlatVector<float>();
    for (size_t i = 0; i < sdf_values_data.size(); ++i) {
        if (sdf_values_data[i] < 0) {
            volumetricPointCloud_->points_.emplace_back(points[3 * i], points[3 * i + 1], points[3 * i + 2]);
        }
    }

    std::cout << "Volumetric point cloud created with " << volumetricPointCloud_->points_.size() << " points." << std::endl;

    return true;
}



bool ModelLoader::generateVoxelGrid(double voxelSize){

    // Check if point cloud data is available
    if (!pointCloud_) {
        std::cerr << "Error: Point cloud data is not available." << std::endl;
        return false;
    }

    // If voxelSize is not provided or less than or equal to zero, set it to twice the average spacing
    if (voxelSize <= 0.0) {
        voxelSize = 2.0 * pointCloudSpacing_;
    }

    // Create a voxel grid representation of the object
    std::cout << "Generating voxel grid with voxel size: " << voxelSize << "..." << std::endl;
    voxelGrid_ = open3d::geometry::VoxelGrid::CreateFromPointCloud(*pointCloud_, voxelSize);
    // voxelGrid_ = open3d::geometry::VoxelGrid::CreateFromTriangleMesh(*meshData_, voxelSize);
    //number of vertices in the mesh



    // Check if the voxel grid is generated successfully
    if (!voxelGrid_->HasVoxels()) {
        std::cerr << "Failed to generate voxel grid." << std::endl;
        return false;
    }

    std::cout << "Voxel grid generated successfully." << std::endl;
     

     // Print voxel grid details
    auto minBound = voxelGrid_->GetMinBound();
    auto maxBound = voxelGrid_->GetMaxBound();
    auto center = voxelGrid_->GetCenter();
    auto voxelCount = voxelGrid_->voxels_.size();
    // Table header
    std::cout << std::setw(40) << std::setfill('=') << "" << std::endl;
    std::cout << std::setfill(' ') << std::left << std::setw(30) << "Voxel Grid Properties" << std::endl;
    std::cout << std::setw(40) << std::setfill('=') << "" << std::setfill(' ') << std::endl;

    // Table content
    std::cout << std::left << std::setw(25) << "Property" << std::setw(15) << "Value" << std::endl;
    std::cout << std::setw(40) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << std::left << std::setw(25) << "Voxel Size:" << std::setw(15) << voxelSize << std::endl;
    std::cout << std::left << std::setw(25) << "Voxel Count:" << std::setw(15) << voxelCount << std::endl;
    std::cout << std::left << std::setw(25) << "Bounding Box Min:" << std::setw(15) << minBound.transpose() << std::endl;
    std::cout << std::left << std::setw(25) << "Bounding Box Max:" << std::setw(15) << maxBound.transpose() << std::endl;
    std::cout << std::left << std::setw(25) << "Bounding Box Center:" << std::setw(15) << center.transpose() << std::endl;
    std::cout << std::setw(40) << std::setfill('=') << "" << std::setfill(' ') << std::endl;

    // Color the voxel grid in red
    for (auto& voxel_pair : voxelGrid_->voxels_) {
        voxel_pair.second.color_ = Eigen::Vector3d(1.0, 0.0, 0.0);  // Red color
    }

    return true;
}


bool ModelLoader::generateOctoMap(double resolution) {
    // Ensure there is point cloud data to work with
    if (!pointCloud_) {
        return false;
    }

    // If resolution is not provided or less than or equal to zero, set it to twice the average point cloud spacing
    if (resolution <= 0.0) {
        resolution = 2.0 * getAverageSpacing();
    }

    // Create an octomap with the specified resolution
    octoMap_ = std::make_shared<octomap::ColorOcTree>(resolution);

    // Iterate through each point in the point cloud
    for (const auto& point : pointCloud_->points_) {
        // Update the octomap node at the point's location and set its color to white (255, 255, 255)
        auto node = octoMap_->updateNode(octomap::point3d(point.x(), point.y(), point.z()), true);
        node->setColor(227, 185, 255);
        // node->setLogOdds(octomap::logodds(1.0)); // Set the log-odds value to 1
    }

    // Update the inner occupancy of the octomap to ensure all nodes reflect occupancy changes
    octoMap_->updateInnerOccupancy();

    std::cout << "Object octomap generated successfully." << std::endl;

    return true;
}



bool ModelLoader::generateVolumetricOctoMap(double resolution) {
    // Ensure volumetric point cloud data is available
    if (!volumetricPointCloud_) {
        std::cerr << "Error: Volumetric point cloud data is not available." << std::endl;
        return false;
    }

    // If resolution is not provided or less than or equal to zero, set it to the default spacing
    if (resolution <= 0.0) {
        resolution = getAverageSpacing();
    }

    // Create an OctoMap with the specified resolution
    volumetricOctomap_ = std::make_shared<octomap::ColorOcTree>(resolution);

    // Iterate through each point in the volumetric point cloud
    for (const auto& point : volumetricPointCloud_->points_) {
        // Update the OctoMap node at the point's location and set its color to a specific value (e.g., white)
        auto node = volumetricOctomap_->updateNode(
            octomap::point3d(point.x(), point.y(), point.z()), true
        );
        node->setColor(128, 128, 128);
        node->setLogOdds(octomap::logodds(1.0)); // Ensure the node is fully occupied
    }

    // Expand all nodes to the specified resolution
    volumetricOctomap_->expand();

    // Update the inner occupancy of the OctoMap to ensure all nodes reflect occupancy changes
    volumetricOctomap_->updateInnerOccupancy();

    std::cout << "Volumetric OctoMap generated successfully." << std::endl;

    return true;
}

bool ModelLoader::generateSurfaceShellOctomap() {
    // Ensure volumetric octomap data is available
    if (!volumetricOctomap_) {
        std::cerr << "Error: Volumetric octomap data is not available." << std::endl;
        return false;
    }

    // Create a new octomap for the surface shell (removed voxels)
    auto surfaceShellOctomap = std::make_shared<octomap::ColorOcTree>(volumetricOctomap_->getResolution());

    // Counter for removed voxels
    int removedVoxelCount = 0;

    // Iterate through each leaf node in the volumetric octomap
    for (auto it = volumetricOctomap_->begin_leafs(); it != volumetricOctomap_->end_leafs(); ++it) {
        if (it->getOccupancy() > 0.5) { // Only consider occupied cells
            octomap::point3d point(it.getX(), it.getY(), it.getZ());
            bool erode = false;

            // Check the 6 neighbors along the principal axes
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dz = -1; dz <= 1; ++dz) {
                        if (std::abs(dx) + std::abs(dy) + std::abs(dz) == 1) {
                            octomap::point3d neighbor = point + octomap::point3d(dx * volumetricOctomap_->getResolution(), dy * volumetricOctomap_->getResolution(), dz * volumetricOctomap_->getResolution());
                            if (!volumetricOctomap_->search(neighbor) || volumetricOctomap_->search(neighbor)->getOccupancy() <= 0.5) {
                                erode = true;
                                break;
                            }
                        }
                    }
                    if (erode) break;
                }
                if (erode) break;
            }

            // If any neighbor is free, add this cell to the surface shell octomap
            if (erode) {
                surfaceShellOctomap->updateNode(point, true);
                surfaceShellOctomap->search(point)->setColor(0, 0, 255); // Set the color to blue
                removedVoxelCount++;
            }
        }
    }

    // Update the inner occupancy of the surface shell octomap to ensure all nodes reflect occupancy changes
    surfaceShellOctomap->updateInnerOccupancy();

    // Store the surface shell octomap for further processing
    surfaceShellOctomap_ = surfaceShellOctomap;

    std::cout << "Surface shell octomap (eroded voxels) generated successfully." << std::endl;
    std::cout << "Number of voxels removed: " << removedVoxelCount << std::endl;

    return true;
}






bool ModelLoader::generateExplorationMap(double resolution, const octomap::point3d& min_bound, const octomap::point3d& max_bound) {
    // Ensure the bounding box is valid
    if (min_bound.x() > max_bound.x() || min_bound.y() > max_bound.y() || min_bound.z() > max_bound.z()) {
        std::cerr << "Invalid bounding box." << std::endl;
        return false;
    }

    // If resolution is not provided or less than or equal to zero, set it to a default value
    if (resolution <= 0.0) {
        resolution = getAverageSpacing();  // Set default resolution if needed
    }

    // Create an exploration map with the specified resolution
    explorationMap_ = std::make_shared<octomap::ColorOcTree>(resolution);

    // Iterate through each point in the bounding box
    for (double x = min_bound.x() + resolution / 2; x <= max_bound.x(); x += resolution) {
        for (double y = min_bound.y() + resolution / 2; y <= max_bound.y(); y += resolution) {
            for (double z = min_bound.z() + resolution / 2; z <= max_bound.z(); z += resolution) {
                // Update the exploration map node at the point's location and set its occupancy to 0.5
                octomap::ColorOcTreeNode* node = explorationMap_->updateNode(octomap::point3d(x, y, z), true);
                if (node) {
                    node->setLogOdds(octomap::logodds(0.5));  // Set occupancy to 0.5
                    node->setColor(255, 255, 0);  // Set color to yellow for visualization
                }
            }
        }
    }

    // Update the inner occupancy of the exploration map to ensure all nodes reflect occupancy changes
    explorationMap_->updateInnerOccupancy();


    std::cout << "Exploration map generated successfully." << std::endl;
    
    return true;
}


bool ModelLoader::generateExplorationMap(int num_cells_per_side, const octomap::point3d& min_bound, const octomap::point3d& max_bound) {
    // Ensure the bounding box is valid
    if (min_bound.x() > max_bound.x() || min_bound.y() > max_bound.y() || min_bound.z() > max_bound.z()) {
        std::cerr << "Invalid bounding box." << std::endl;
        return false;
    }

    // Calculate the maximum side length to form a cube
    double max_side_length = std::max({max_bound.x() - min_bound.x(), max_bound.y() - min_bound.y(), max_bound.z() - min_bound.z()});

    // Calculate the resolution based on the number of cells per side
    double resolution = max_side_length / num_cells_per_side;
    octomap_resolution_ = resolution;

    // Center the cube around the center of the original bounding box
    octomap::point3d center = (min_bound + max_bound) * 0.5;
    octomap::point3d new_min_bound = center - octomap::point3d(max_side_length / 2, max_side_length / 2, max_side_length / 2);
    octomap::point3d new_max_bound = center + octomap::point3d(max_side_length / 2, max_side_length / 2, max_side_length / 2);

    // Create an exploration map with the specified resolution
    explorationMap_ = std::make_shared<octomap::ColorOcTree>(resolution);

    // Iterate through each point in the new bounding box
    for (double x = new_min_bound.x() + resolution / 2; x <= new_max_bound.x(); x += resolution) {
        for (double y = new_min_bound.y() + resolution / 2; y <= new_max_bound.y(); y += resolution) {
            for (double z = new_min_bound.z() + resolution / 2; z <= new_max_bound.z(); z += resolution) {
                // Update the exploration map node at the point's location and set its occupancy to 0.5
                octomap::ColorOcTreeNode* node = explorationMap_->updateNode(octomap::point3d(x, y, z), true);
                if (node) {
                    node->setLogOdds(octomap::logodds(0.5));  // Set occupancy to 0.5
                    node->setColor(255, 255, 0);  // Set color to yellow for visualization
                }
            }
        }
    }

    // Update the inner occupancy of the exploration map to ensure all nodes reflect occupancy changes
    explorationMap_->updateInnerOccupancy();

    std::cout << "Exploration map generated successfully." << std::endl;

    return true;
}




} // namespace visioncraft
