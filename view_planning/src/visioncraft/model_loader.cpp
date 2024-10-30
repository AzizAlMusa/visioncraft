#include "visioncraft/model_loader.h" // Include the ModelLoader header file

#include <iostream>
#include <iomanip>
#include <string>
#include <Eigen/Core>
#include <chrono>


namespace visioncraft {

// Constructor for the ModelLoader class
ModelLoader::ModelLoader() {

}

// Destructor for the ModelLoader class
ModelLoader::~ModelLoader() {
    if (gpu_voxel_grid_.voxel_data) {
        delete[] gpu_voxel_grid_.voxel_data;
        gpu_voxel_grid_.voxel_data = nullptr;
    }

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
        // std::cout << "Mesh loaded successfully with " << meshData_->vertices_.size() << " vertices." << std::endl;
    
        auto aabb = meshData_->GetAxisAlignedBoundingBox();
        // std::cout << "Bounding box: min = " << aabb.min_bound_.transpose() << ", max = " << aabb.max_bound_.transpose() << ", center = " << aabb.GetCenter().transpose() << ", extent = " << aabb.GetExtent().transpose() << std::endl;

         // Assign the bounding box information to the member variables
        minBound_ = octomap::point3d(aabb.min_bound_.x(), aabb.min_bound_.y(), aabb.min_bound_.z());
        maxBound_ = octomap::point3d(aabb.max_bound_.x(), aabb.max_bound_.y(), aabb.max_bound_.z());
        center_ = octomap::point3d(aabb.GetCenter().x(), aabb.GetCenter().y(), aabb.GetCenter().z());

        return true;
    }
}

// Load a 3D model and generate all necessary structures
bool ModelLoader::loadModel(const std::string& file_path, int num_samples, double resolution) {
    // std::cout << "===================== MODEL LOADER ========================" << std::endl;
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

    // Lambda to measure the time taken by each function call
    auto measureTime = [](auto func, const std::string& func_name) {
        auto start = std::chrono::high_resolution_clock::now();
        bool result = func();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << func_name << " took " << elapsed.count() << " seconds." << std::endl;
        return result;
    };

    // Measure each function call
    success &= measureTime([this]() { return initializeRaycastingScene(); }, "initializeRaycastingScene");
    // success &= measureTime([this, num_samples]() { return generatePointCloud(num_samples); }, "generatePointCloud");

    if (resolution <= 0) {
        // resolution = 8.0 * getAverageSpacing();

        // // Get bounding box of the mesh
        auto aabb = meshData_->GetAxisAlignedBoundingBox();
        double extent_max = aabb.GetExtent().maxCoeff();  // Get the longest side of the bounding box
        int voxel_count_target = 24;  // Target number of voxels per side for reasonable detail

        // Estimate resolution based on the longest side and target voxel count
        resolution = extent_max / voxel_count_target;
        pointCloudSpacing_ = resolution;
        // std::cout << "Using default resolution: " << resolution << std::endl;
    }

    success &= measureTime([this, resolution]() { return generateOctoMapFromMesh(resolution); }, "generateOctoMapFromMesh");
    // success &= measureTime([this, resolution]() { return generateOctoMap(resolution); }, "generateOctoMap");
    success &= measureTime([this, resolution]() { return generateVolumetricOctoMap(resolution); }, "generateVolumetricOctoMap");
    success &= measureTime([this]() { return generateSurfaceShellOctomap(); }, "generateSurfaceShellOctomap");
    success &= measureTime([this, resolution]() { return convertVoxelGridToGPUFormat(resolution); }, "convertVoxelGridToGPUFormat");
    success &= measureTime([this, resolution]() { return generateExplorationMap(resolution, getMinBound(), getMaxBound()); }, "generateExplorationMap");

    // std::cout << "===================== MODEL LOADER COMPLETE ========================" << std::endl;

    return success;
}

bool ModelLoader::generateExplorationStructures(int num_samples, int num_cells_per_side) {
    bool success = true;

    success &= initializeRaycastingScene();
    success &= generatePointCloud(num_samples); // Number of samples for the point cloud
    success &= generateExplorationMap(num_cells_per_side, getMinBound(), getMaxBound());

    double resolution = getExplorationMapResolution();
    success &= generateOctoMap(resolution);
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
    // std::cout << "Mesh Data Information:" << std::endl;
    // std::cout << "Number of Vertices: " << meshData_->vertices_.size() << std::endl;
    // std::cout << "Number of Triangles: " << meshData_->triangles_.size() << std::endl;


  
    // Generate point cloud using Poisson disk sampling from Open3D
    // std::cout << "Generating point cloud using Poisson disk sampling with " << numSamples << " samples..."  << std::endl;
    // std::shared_ptr<open3d::geometry::PointCloud> pcd = meshData_->SamplePointsPoissonDisk(numSamples, 5.0, nullptr, true);
    // std::cout << "Generating point cloud using Uniform sampling with " << numSamples << " samples..."  << std::endl;
    std::shared_ptr<open3d::geometry::PointCloud> pcd = meshData_->SamplePointsUniformly(numSamples, true);
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
    // std::cout << "Generating normals..."  << std::endl ;
    pcd->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(averageSpacing * 5, 30));  // Adjust these parameters as needed

    // Correct oppositely oriented normals using signed distance
    // std::cout << "Correcting misaligned normals..."  << std::endl ;
    pcd = correctNormalsUsingSignedDistance(pcd, averageSpacing * 0.01);


    // Store the generated point cloud
    pointCloud_ = pcd;

    // std::cout << "Point cloud created successfully."  << std::endl ;
    
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



bool ModelLoader::generateVoxelGrid(double voxelSize){

    // Check if point cloud data is available
    if (!pointCloud_) {
        std::cerr << "Error: Point cloud data is not available." << std::endl;
        return false;
    }

    // If voxelSize is not provided or less than or equal to zero, set it to twice the average spacing
    if (voxelSize <= 0.0) {
        voxelSize = 10.0 * getAverageSpacing();
    }

    // Create a voxel grid representation of the object
    // std::cout << "Generating voxel grid with voxel size: " << voxelSize << "..." << std::endl;
    voxelGrid_ = open3d::geometry::VoxelGrid::CreateFromPointCloud(*pointCloud_, voxelSize);
    // voxelGrid_ = open3d::geometry::VoxelGrid::CreateFromTriangleMesh(*meshData_, voxelSize);
    //number of vertices in the mesh



    // Check if the voxel grid is generated successfully
    if (!voxelGrid_->HasVoxels()) {
        std::cerr << "Failed to generate voxel grid." << std::endl;
        return false;
    }

    // std::cout << "Voxel grid generated successfully." << std::endl;
     

     // Print voxel grid details
    auto minBound = voxelGrid_->GetMinBound();
    auto maxBound = voxelGrid_->GetMaxBound();
    auto center = voxelGrid_->GetCenter();
    auto voxelCount = voxelGrid_->voxels_.size();
    // // Table header
    // std::cout << std::setw(40) << std::setfill('=') << "" << std::endl;
    // std::cout << std::setfill(' ') << std::left << std::setw(30) << "Voxel Grid Properties" << std::endl;
    // std::cout << std::setw(40) << std::setfill('=') << "" << std::setfill(' ') << std::endl;

    // // Table content
    // std::cout << std::left << std::setw(25) << "Property" << std::setw(15) << "Value" << std::endl;
    // std::cout << std::setw(40) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    // std::cout << std::left << std::setw(25) << "Voxel Size:" << std::setw(15) << voxelSize << std::endl;
    // std::cout << std::left << std::setw(25) << "Voxel Count:" << std::setw(15) << voxelCount << std::endl;
    // std::cout << std::left << std::setw(25) << "Bounding Box Min:" << std::setw(15) << minBound.transpose() << std::endl;
    // std::cout << std::left << std::setw(25) << "Bounding Box Max:" << std::setw(15) << maxBound.transpose() << std::endl;
    // std::cout << std::left << std::setw(25) << "Bounding Box Center:" << std::setw(15) << center.transpose() << std::endl;
    // std::cout << std::setw(40) << std::setfill('=') << "" << std::setfill(' ') << std::endl;

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
        resolution = 10.0 * getAverageSpacing();
    }

    // Create an octomap with the specified resolution
    octoMap_ = std::make_shared<octomap::ColorOcTree>(resolution);

    // Iterate through each point in the point cloud
    for (const auto& point : pointCloud_->points_) {
        // Update the octomap node at the point's location and set its color to white (255, 255, 255)
        auto node = octoMap_->updateNode(octomap::point3d(point.x(), point.y(), point.z()), true);
        node->setColor(255, 255, 255);
        // node->setLogOdds(octomap::logodds(1.0)); // Set the log-odds value to 1
    }

    // Update the inner occupancy of the octomap to ensure all nodes reflect occupancy changes
    octoMap_->updateInnerOccupancy();

    // std::cout << "Object octomap generated successfully." << std::endl;

    return true;
}


bool ModelLoader::generateOctoMapFromMesh(double resolution) {
    // Ensure mesh data is loaded
    if (!meshData_) {
        std::cerr << "Error: Mesh data is not loaded." << std::endl;
        return false;
    }

    // Set resolution for the octomap if not specified
    if (resolution <= 0.0) {
       

        // Get bounding box of the mesh
        auto aabb = meshData_->GetAxisAlignedBoundingBox();
        double extent_max = aabb.GetExtent().maxCoeff();  // Get the longest side of the bounding box
        int voxel_count_target = 100;  // Target number of voxels per side for reasonable detail

        // Estimate resolution based on the longest side and target voxel count
        resolution = extent_max / voxel_count_target;
        
        pointCloudSpacing_ = resolution;

    }

    // Create a voxel grid from the mesh
    auto voxel_grid = open3d::geometry::VoxelGrid::CreateFromTriangleMesh(*meshData_, resolution);
    if (!voxel_grid->HasVoxels()) {
        std::cerr << "Error: Failed to generate voxel grid from mesh." << std::endl;
        return false;
    }

    // Initialize the octomap with the same resolution
    octoMap_ = std::make_shared<octomap::ColorOcTree>(resolution);

    // Iterate over each voxel in the voxel grid and add it to the octomap
    for (const auto& voxel : voxel_grid->voxels_) {
        // Get the voxel center as the insertion point in octomap
        const auto& center = voxel_grid->GetVoxelCenterCoordinate(voxel.first);
        octomap::point3d point(center.x(), center.y(), center.z());

        // Update node in octomap at voxel center and set it as occupied
        auto node = octoMap_->updateNode(point, true);
        node->setColor(255, 255, 255);  // Set color to white for visualization
    }

    // Update inner occupancy of the octomap for accurate occupancy reflections
    octoMap_->updateInnerOccupancy();

    // Optional: Log octomap generation
    // std::cout << "Octomap generated from mesh with resolution: " << resolution << std::endl;

    return true;
}

bool ModelLoader::generateVolumetricOctoMap(double resolution) {
    // Ensure raycasting scene and octoMap_ are initialized
    if (!raycasting_scene_ || !octoMap_) {
        std::cerr << "Error: Raycasting scene or octoMap_ is not initialized." << std::endl;
        return false;
    }

    // Initialize volumetricOctomap_ as a copy of octoMap_
    volumetricOctomap_ = std::make_shared<octomap::ColorOcTree>(*octoMap_);

    // Get bounding box min and max bounds from octoMap_ using getMetricMin and getMetricMax
    double min_x, min_y, min_z, max_x, max_y, max_z;
    octoMap_->getMetricMin(min_x, min_y, min_z);
    octoMap_->getMetricMax(max_x, max_y, max_z);

    // Use octoMap_ resolution for consistent voxel spacing
    double step = octoMap_->getResolution();

    // Prepare flat vector for batched SDF query
    std::vector<float> batch_points;
    std::vector<octomap::point3d> voxel_positions;

    // Updated loop with offset (step / 2) for centering
    for (double x = min_x + step / 2; x <= max_x; x += step) {
        for (double y = min_y + step / 2; y <= max_y; y += step) {
            for (double z = min_z + step / 2; z <= max_z; z += step) {
                // Store x, y, z as floats in the flat vector
                batch_points.push_back(static_cast<float>(x));
                batch_points.push_back(static_cast<float>(y));
                batch_points.push_back(static_cast<float>(z));

                // Save the voxel positions for later use
                voxel_positions.emplace_back(x, y, z);
            }
        }
    }

    // Create Open3D tensor from flat data
    open3d::core::Tensor points_tensor(
        batch_points, {static_cast<int64_t>(batch_points.size() / 3), 3}, 
        open3d::core::Dtype::Float32, open3d::core::Device("CPU:0"));

    // Compute signed distances for the batch of points
    open3d::core::Tensor sdf_values = raycasting_scene_->ComputeSignedDistance(points_tensor);
    auto sdf_values_data = sdf_values.ToFlatVector<float>();

    // Populate the volumetricOctomap_ based on the signed distance values
    for (size_t i = 0; i < sdf_values_data.size(); ++i) {
        if (sdf_values_data[i] < 0) { // Inside the object
            volumetricOctomap_->updateNode(voxel_positions[i], true);
        }
    }

    // Update inner occupancy to finalize the octomap
    volumetricOctomap_->updateInnerOccupancy();
    // std::cout << "Volumetric OctoMap generated successfully." << std::endl;

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

    // std::cout << "Surface shell octomap (eroded voxels) generated successfully." << std::endl;
    // std::cout << "Number of voxels removed: " << removedVoxelCount << std::endl;

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


    // std::cout << "Exploration map generated successfully." << std::endl;
    
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

    // std::cout << "Exploration map generated successfully." << std::endl;

    return true;
}

/**
 * @brief Convert the voxel grid to a GPU-friendly structure for efficient raycasting.
 * 
 * This function converts the current voxel grid into a format that can be easily
 * transferred to the GPU. The grid is stored as a linearized 1D array of occupancy 
 * data (0 = free, 1 = occupied) and includes dimensions and voxel size.
 * 
 * @param voxelSize The size of each voxel in the grid.
 * @return True if the conversion is successful, false otherwise.
 */
bool ModelLoader::convertVoxelGridToGPUFormat(double voxelSize) {
    // Ensure surfaceShellOctomap_ data is available
    if (!surfaceShellOctomap_) {
        std::cerr << "Error: Surface shell OctoMap data is not available." << std::endl;
        return false;
    }

    // Get OctoMap bounds from the surface shell OctoMap
    double min_x, min_y, min_z, max_x, max_y, max_z;
    surfaceShellOctomap_->getMetricMin(min_x, min_y, min_z);
    surfaceShellOctomap_->getMetricMax(max_x, max_y, max_z);

    gpu_voxel_grid_.voxel_size = voxelSize;

    // Calculate dimensions of the voxel grid
    gpu_voxel_grid_.width = static_cast<int>((max_x - min_x) / voxelSize);
    gpu_voxel_grid_.height = static_cast<int>((max_y - min_y) / voxelSize);
    gpu_voxel_grid_.depth = static_cast<int>((max_z - min_z) / voxelSize);

    // Set min_bound in gpu_voxel_grid_
    gpu_voxel_grid_.min_bound[0] = static_cast<float>(min_x) + voxelSize / 2;
    gpu_voxel_grid_.min_bound[1] = static_cast<float>(min_y) + voxelSize / 2;
    gpu_voxel_grid_.min_bound[2] = static_cast<float>(min_z) + voxelSize / 2;

    // Calculate total number of voxels and allocate memory
    int total_voxels = gpu_voxel_grid_.width * gpu_voxel_grid_.height * gpu_voxel_grid_.depth;
    gpu_voxel_grid_.voxel_data = new int[total_voxels];
    std::fill(gpu_voxel_grid_.voxel_data, gpu_voxel_grid_.voxel_data + total_voxels, 0);

    // Populate the voxel data array directly from surfaceShellOctomap_ leaf nodes
    for (auto it = surfaceShellOctomap_->begin_leafs(); it != surfaceShellOctomap_->end_leafs(); ++it) {
        auto voxel_center = it.getCoordinate();

        // Calculate voxel indices (x, y, z) based on the voxel center position
        int x_idx = static_cast<int>((voxel_center.x() - min_x) / voxelSize);
        int y_idx = static_cast<int>((voxel_center.y() - min_y) / voxelSize);
        int z_idx = static_cast<int>((voxel_center.z() - min_z) / voxelSize);

        // Calculate the linear index for the voxel in the 1D array
        int linear_idx = z_idx * (gpu_voxel_grid_.width * gpu_voxel_grid_.height) + y_idx * gpu_voxel_grid_.width + x_idx;

        if (linear_idx >= 0 && linear_idx < total_voxels) {
            gpu_voxel_grid_.voxel_data[linear_idx] = 1;
        }
    }

    return true;
}


/**
 * @brief Update the voxel grid based on the hit voxels.
 * 
 * This function takes a set of 3D voxel indices (x, y, z) representing hit voxels and
 * updates the corresponding voxels in the GPU-friendly voxel grid to occupied (value = 1).
 * 
 * @param unique_hit_voxels A set of 3D voxel indices (x, y, z) representing hit voxels.
 */
void ModelLoader::updateVoxelGridFromHits(const std::set<std::tuple<int, int, int>>& unique_hit_voxels) {
    for (const auto& voxel_idx : unique_hit_voxels) {
        int x = std::get<0>(voxel_idx);
        int y = std::get<1>(voxel_idx);
        int z = std::get<2>(voxel_idx);

        // Compute the linear index from (x, y, z)
        int linear_idx = z * (gpu_voxel_grid_.width * gpu_voxel_grid_.height) + y * gpu_voxel_grid_.width + x;

        // Ensure the index is within bounds before updating the voxel data
        if (linear_idx >= 0 && linear_idx < gpu_voxel_grid_.width * gpu_voxel_grid_.height * gpu_voxel_grid_.depth) {
            gpu_voxel_grid_.voxel_data[linear_idx] = 1;  // Mark as occupied
        } else {
            std::cerr << "Error: Voxel index out of bounds!" << std::endl;
        }
    }
}

/**
 * @brief Update the OctoMap based on the hit voxels.
 * 
 * This function takes a set of 3D voxel indices (x, y, z), converts them to world coordinates,
 * and updates the corresponding voxels in the OctoMap to have a green color (0, 255, 0).
 * 
 * @param unique_hit_voxels A set of 3D voxel indices (x, y, z) representing hit voxels.
 */
void ModelLoader::updateOctomapWithHits(const std::set<std::tuple<int, int, int>>& unique_hit_voxels) {
    for (const auto& voxel_idx : unique_hit_voxels) {
        int x = std::get<0>(voxel_idx);
        int y = std::get<1>(voxel_idx);
        int z = std::get<2>(voxel_idx);

        // Calculate the world coordinates of the voxel center
        double x_world = gpu_voxel_grid_.min_bound[0] + x * gpu_voxel_grid_.voxel_size;
        double y_world = gpu_voxel_grid_.min_bound[1] + y * gpu_voxel_grid_.voxel_size;
        double z_world = gpu_voxel_grid_.min_bound[2] + z * gpu_voxel_grid_.voxel_size;

        // Update the voxel in the OctoMap and set its color to green (0, 255, 0)
        octomap::ColorOcTreeNode* node = surfaceShellOctomap_->updateNode(octomap::point3d(x_world, y_world, z_world), true);
        if (node) {
            // std::cout << "Updating OctoMap node at (" << x_world << ", " << y_world << ", " << z_world << ")" << std::endl;
            node->setColor(0, 255, 0);  // Green color for hit voxels
        } else {
            std::cerr << "Error: Failed to update OctoMap node!" << std::endl;
        }
    }
}


/**
 * @brief Generate a map of MetaVoxel instances by copying the structure from the surface shell octomap.
 * 
 * This function iterates over each leaf node in the surface shell octomap and creates a MetaVoxel
 * object for each occupied voxel. Each MetaVoxel is stored in the meta voxel map, keyed by the 
 * OctoMap key of the corresponding voxel.
 * 
 * @return True if the meta voxel map is generated successfully, false otherwise.
 */
bool ModelLoader::generateVoxelMap() {
    if (!surfaceShellOctomap_) {
        std::cerr << "Error: Surface shell OctoMap is not available." << std::endl;
        return false;
    }

    // Clear any existing data in the meta voxel map
    meta_voxel_map_.clear();

    // Iterate through each leaf node in the surface shell octomap
    for (auto it = surfaceShellOctomap_->begin_leafs(); it != surfaceShellOctomap_->end_leafs(); ++it) {
        if (it->getOccupancy() > 0.5) { // Only consider occupied cells
            octomap::OcTreeKey key = it.getKey();
            Eigen::Vector3d position(it.getX(), it.getY(), it.getZ());
            float occupancy = it->getOccupancy();
            MetaVoxel meta_voxel(position, key, occupancy);

            // Insert the MetaVoxel into the map
            meta_voxel_map_.setMetaVoxel(key, meta_voxel);
        }
    }

    // std::cout << "Meta voxel map generated successfully with " << meta_voxel_map_.size() << " voxels." << std::endl;
    return true;
}

/**
 * @brief Retrieve a MetaVoxel object from the meta voxel map using the specified OctoMap key.
 * 
 * This function provides efficient access to the MetaVoxel object associated with the provided key.
 * 
 * @param key The OctoMap key of the MetaVoxel to retrieve.
 * @return Pointer to the MetaVoxel if found, nullptr otherwise.
 */
MetaVoxel* ModelLoader::getVoxel(const octomap::OcTreeKey& key) {
    return meta_voxel_map_.getMetaVoxel(key);
}

/**
 * @brief Retrieve a MetaVoxel object from the meta voxel map using the specified voxel position.
 * 
 * This function converts the 3D voxel position to an OctoMap key and then retrieves the corresponding MetaVoxel.
 * 
 * @param position The 3D position of the voxel.
 * @return Pointer to the MetaVoxel if found, nullptr otherwise.
 */
MetaVoxel* ModelLoader::getVoxel(const Eigen::Vector3d& position) {
    octomap::OcTreeKey key;
    if (surfaceShellOctomap_ && surfaceShellOctomap_->coordToKeyChecked(octomap::point3d(position.x(), position.y(), position.z()), key)) {
        return meta_voxel_map_.getMetaVoxel(key);
    }
    std::cerr << "Failed to convert position to OctoMap key." << std::endl;
    return nullptr;
}

/**
 * @brief Update the occupancy value of a MetaVoxel in the meta voxel map.
 * 
 * This function allows modification of the occupancy value of a MetaVoxel, identified by its OctoMap key,
 * and updates the log-odds of occupancy accordingly.
 * 
 * @param key The OctoMap key of the MetaVoxel to update.
 * @param new_occupancy The new occupancy probability for the MetaVoxel.
 * @return True if the MetaVoxel is updated successfully, false otherwise.
 */
bool ModelLoader::updateVoxelOccupancy(const octomap::OcTreeKey& key, float new_occupancy) {
    MetaVoxel* voxel = getVoxel(key);
    if (voxel) {
        voxel->setOccupancy(new_occupancy);
        return true;
    }
    return false;
}

/**
 * @brief Update the occupancy value of a MetaVoxel in the meta voxel map using the specified voxel position.
 * 
 * This function converts the 3D voxel position to an OctoMap key and then updates the occupancy of the corresponding MetaVoxel.
 * 
 * @param position The 3D position of the voxel.
 * @param new_occupancy The new occupancy probability for the MetaVoxel.
 * @return True if the MetaVoxel is updated successfully, false otherwise.
 */
bool ModelLoader::updateVoxelOccupancy(const Eigen::Vector3d& position, float new_occupancy) {
    MetaVoxel* voxel = getVoxel(position);
    if (voxel) {
        voxel->setOccupancy(new_occupancy);
        return true;
    }
    std::cerr << "MetaVoxel at the specified position not found." << std::endl;
    return false;
}


bool ModelLoader::addVoxelProperty(const std::string& property_name, const MetaVoxel::PropertyValue& initial_value) {
    return meta_voxel_map_.setPropertyForAllVoxels(property_name, initial_value);
}



/**
 * @brief Set a custom property for a MetaVoxel identified by its OctoMap key.
 * 
 * This function allows adding or updating a custom property for a MetaVoxel within the meta voxel map.
 * 
 * @param key The OctoMap key of the MetaVoxel to update.
 * @param property_name The name of the property to set.
 * @param value The value to assign to the property.
 * @return True if the property is set successfully, false otherwise.
 */
bool ModelLoader::setVoxelProperty(const octomap::OcTreeKey& key, const std::string& property_name, const MetaVoxel::PropertyValue& value) {
    return meta_voxel_map_.setMetaVoxelProperty(key, property_name, value);
}


/**
 * @brief Set a custom property for a MetaVoxel identified by its voxel position.
 * 
 * This function converts the 3D voxel position to an OctoMap key and then sets the specified property in the corresponding MetaVoxel.
 * 
 * @param position The 3D position of the voxel.
 * @param property_name The name of the property to set.
 * @param value The value to assign to the property.
 * @return True if the property is set successfully, false otherwise.
 */
bool ModelLoader::setVoxelProperty(const Eigen::Vector3d& position, const std::string& property_name, const MetaVoxel::PropertyValue& value) {
    MetaVoxel* voxel = getVoxel(position);
    if (voxel) {
        voxel->setProperty(property_name, value);
        return true;
    }
    std::cerr << "Failed to set property for MetaVoxel at the specified position." << std::endl;
    return false;
}

/**
 * @brief Retrieve a custom property from a MetaVoxel.
 * 
 * This function retrieves the value of a specified property from a MetaVoxel if the property exists.
 * 
 * @param key The OctoMap key of the MetaVoxel.
 * @param property_name The name of the property to retrieve.
 * @return The value of the property if found, throws runtime_error if the MetaVoxel or property is not found.
 */
MetaVoxel::PropertyValue ModelLoader::getVoxelProperty(const octomap::OcTreeKey& key, const std::string& property_name) const {
    return meta_voxel_map_.getMetaVoxelProperty(key, property_name);
}

/**
 * @brief Retrieve a custom property from a MetaVoxel using its voxel position.
 * 
 * This function converts the 3D voxel position to an OctoMap key and then retrieves the specified property from the corresponding MetaVoxel.
 * 
 * @param position The 3D position of the voxel.
 * @param property_name The name of the property to retrieve.
 * @return The value of the property if found, throws runtime_error if the MetaVoxel or property is not found.
 */
MetaVoxel::PropertyValue ModelLoader::getVoxelProperty(const Eigen::Vector3d& position, const std::string& property_name) const {
    octomap::OcTreeKey key;
    if (surfaceShellOctomap_ && surfaceShellOctomap_->coordToKeyChecked(octomap::point3d(position.x(), position.y(), position.z()), key)) {
        return meta_voxel_map_.getMetaVoxelProperty(key, property_name);
    }
    throw std::runtime_error("Failed to retrieve property for MetaVoxel at the specified position.");
}

} // namespace visioncraft
