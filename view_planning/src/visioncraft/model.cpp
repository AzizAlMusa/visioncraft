#include "visioncraft/model.h" // Include the Model header file

#include <iostream>
#include <fstream>
#include <sstream>

#include <iomanip>
#include <string>
#include <Eigen/Core>
#include <chrono>


namespace visioncraft {

// Constructor for the Model class
Model::Model() {

}

// Destructor for the Model class
Model::~Model() {
    if (gpu_voxel_grid_.voxel_data) {
        delete[] gpu_voxel_grid_.voxel_data;
        gpu_voxel_grid_.voxel_data = nullptr;
    }

}

// Load a mesh from a file
bool Model::loadMesh(const std::string& file_path) {
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


bool Model::loadBinvoxToOctomap(const std::string& file_path) {
    std::ifstream ifs(file_path, std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "Error: Could not open binvox file " << file_path << std::endl;
        return false;
    }

    // Variables for parsing the header
    std::string line;
    int width = 0, height = 0, depth = 0;
    double resolution = 0.0;
    double tx = 0.0, ty = 0.0, tz = 0.0;
    bool has_header = false;

    // Parse the header
    while (std::getline(ifs, line)) {
        if (line.substr(0, 7) == "#binvox") {
            has_header = true;
        } else if (line.substr(0, 3) == "dim") {
            std::istringstream dims(line.substr(4));
            dims >> width >> height >> depth;
        } else if (line.substr(0, 9) == "translate") {
            std::istringstream trans(line.substr(10));
            trans >> tx >> ty >> tz;
        } else if (line.substr(0, 5) == "scale") {
            std::istringstream scale(line.substr(6));
            scale >> resolution;
        } else if (line == "data") {
            break;
        }
    }

    // Validate header data
    if (!has_header || width == 0 || height == 0 || depth == 0 || resolution <= 0.0) {
        std::cerr << "Error: Invalid or incomplete binvox header in " << file_path << std::endl;
        return false;
    }

    // Prepare the OctoMap with the parsed resolution
    octoMap_ = std::make_shared<octomap::ColorOcTree>(resolution);

    // Read voxel data
    unsigned char value;
    unsigned char count;
    int index = 0;

    while (ifs.read((char*)&value, 1) && ifs.read((char*)&count, 1)) {
        for (int i = 0; i < count; ++i, ++index) {
            int x = index % width;
            int y = (index / width) % height;
            int z = index / (width * height);

            if (value == 1) { // Mark voxel as occupied if value is 1
                octomap::point3d voxel_center(
                    tx + x * resolution,
                    ty + y * resolution,
                    tz + z * resolution
                );
                octoMap_->updateNode(voxel_center, true);
            }
        }
    }

    ifs.close();

    // Count and print the number of leaf nodes
    size_t leaf_count = 0;
    for (auto it = octoMap_->begin_leafs(), end = octoMap_->end_leafs(); it != end; ++it) {
        ++leaf_count;
    }
    std::cout << "Total leaf nodes in OctoMap: " << leaf_count << std::endl;

    return true;
}


// Load a 3D model and generate all necessary structures
bool Model::loadModel(const std::string& file_path, int num_samples, double resolution) {
    
    clear();  // Clear any existing data before loading a new model

    if (!loadMesh(file_path)) {
        return false;
    }
    return generateAllStructures(num_samples, resolution);
}

// Load a 3D model and generate all necessary structures for exploration
bool Model::loadExplorationModel(const std::string& file_path, int num_samples, int num_cells_per_side ){
    clear();  // Clear any existing data before loading a new model

    if (!loadMesh(file_path)) {
        return false;
    }
    return generateExplorationStructures(num_samples, num_cells_per_side);
}

// Generate all necessary structures from the loaded mesh
bool Model::generateAllStructures(int num_samples, double resolution) {
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

    // Original function calls without measureTime
    success &= initializeRaycastingScene();
    success &= generatePointCloud(num_samples);

    // success &= loadBinvoxToOctomap("../models/model_normalized.surface.binvox");
    if (resolution <= 0) {
        resolution = 8.0 * getAverageSpacing();
        voxel_size_ = resolution;
    }

    success &= generateOctoMap(resolution);
    success &= generateVolumetricOctoMap(resolution);
    success &= generateSurfaceShellOctomap();
    success &= convertVoxelGridToGPUFormat(resolution);
    success &= generateVoxelMap();
    success &= generateExplorationMap(resolution, getMinBound(), getMaxBound());

    return success;
}


bool Model::generateExplorationStructures(int num_samples, int num_cells_per_side) {
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
bool Model::initializeRaycastingScene() {
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
bool Model::generatePointCloud(int numSamples) {
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




std::shared_ptr<open3d::geometry::PointCloud> Model::correctNormalsUsingSignedDistance(std::shared_ptr<open3d::geometry::PointCloud> pointcloud, double epsilon) {
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



bool Model::generateVoxelGrid(double voxelSize){

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


bool Model::generateOctoMap(double resolution) {
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


bool Model::generateOctoMapFromMesh(double resolution) {
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

bool Model::generateVolumetricOctoMap(double resolution) {
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




bool Model::generateSurfaceShellOctomap() {
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






bool Model::generateExplorationMap(double resolution, const octomap::point3d& min_bound, const octomap::point3d& max_bound) {
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


bool Model::generateExplorationMap(int num_cells_per_side, const octomap::point3d& min_bound, const octomap::point3d& max_bound) {
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
bool Model::convertVoxelGridToGPUFormat(double voxelSize) {
    if (!surfaceShellOctomap_) {
        std::cerr << "Error: Surface shell OctoMap data is not available." << std::endl;
        return false;
    }

    // Get OctoMap bounds from the surface shell OctoMap
    double min_x, min_y, min_z, max_x, max_y, max_z;
    surfaceShellOctomap_->getMetricMin(min_x, min_y, min_z);
    surfaceShellOctomap_->getMetricMax(max_x, max_y, max_z);

    // Adjust max bounds to the nearest lower multiple of voxelSize from min bound
    max_x = min_x + std::floor((max_x - min_x) / voxelSize) * voxelSize;
    max_y = min_y + std::floor((max_y - min_y) / voxelSize) * voxelSize;
    max_z = min_z + std::floor((max_z - min_z) / voxelSize) * voxelSize;

    gpu_voxel_grid_.voxel_size = voxelSize;

    // Calculate dimensions of the voxel grid
    gpu_voxel_grid_.width = static_cast<int>((max_x - min_x) / voxelSize) + 1;
    gpu_voxel_grid_.height = static_cast<int>((max_y - min_y) / voxelSize) + 1;
    gpu_voxel_grid_.depth = static_cast<int>((max_z - min_z) / voxelSize) + 1;

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

        // Ensure indices are within calculated dimensions
        if (x_idx >= 0 && x_idx < gpu_voxel_grid_.width &&
            y_idx >= 0 && y_idx < gpu_voxel_grid_.height &&
            z_idx >= 0 && z_idx < gpu_voxel_grid_.depth) {

            // Calculate the linear index for the voxel in the 1D array
            int linear_idx = z_idx * (gpu_voxel_grid_.width * gpu_voxel_grid_.height) + y_idx * gpu_voxel_grid_.width + x_idx;

            // Mark voxel as occupied
            gpu_voxel_grid_.voxel_data[linear_idx] = 1;
        } else {
            // Debugging output for out-of-bounds voxels
            std::cerr << "Voxel center (" << voxel_center.x() << ", " << voxel_center.y() << ", " << voxel_center.z()
                      << ") mapped to out-of-bounds indices (" << x_idx << ", " << y_idx << ", " << z_idx << ")" << std::endl;
        }
    }

    return true;
}


/**
 * @brief Convert hit voxels from GPU format to OctoMap format.
 * 
 * This function takes a set of hit voxel indices (x, y, z) from the GPU raycasting results,
 * converts each index into world coordinates, and then generates an OctoMap key for each
 * voxel. The resulting map has keys representing each voxel's OctoMap coordinates and
 * a value of `true` to indicate that the voxel was hit.
 * 
 * @param unique_hit_voxels A set of 3D voxel indices (x, y, z) representing hit voxels.
 * @return A map where the key is the voxel key (octomap::OcTreeKey), and the value is a boolean indicating if the voxel was hit.
 */
std::unordered_map<octomap::OcTreeKey, bool, octomap::OcTreeKey::KeyHash> Model::convertGPUHitsToOctreeKeys(
    const std::set<std::tuple<int, int, int>>& unique_hit_voxels) const {

    std::unordered_map<octomap::OcTreeKey, bool, octomap::OcTreeKey::KeyHash> octree_hits;

    // Min bound and voxel size to match the setup in convertVoxelGridToGPUFormat
    double min_x, min_y, min_z;
    surfaceShellOctomap_->getMetricMin(min_x, min_y, min_z);
    double voxelSize = gpu_voxel_grid_.voxel_size;

    for (const auto& voxel_idx : unique_hit_voxels) {
        int x = std::get<0>(voxel_idx);
        int y = std::get<1>(voxel_idx);
        int z = std::get<2>(voxel_idx);

        // Calculate world coordinates, adjusted to the center of each voxel
        double x_world = min_x + x * voxelSize + voxelSize / 2.0;
        double y_world = min_y + y * voxelSize + voxelSize / 2.0;
        double z_world = min_z + z * voxelSize + voxelSize / 2.0;

        // Convert world coordinates to OctoMap key
        octomap::point3d world_point(x_world, y_world, z_world);
        octomap::OcTreeKey key = surfaceShellOctomap_->coordToKey(world_point);

        // Insert into the map with a value of true to indicate a hit
        octree_hits[key] = true;
    }

    return octree_hits;
}


/**
 * @brief Update the voxel grid based on the hit voxels.
 * 
 * This function takes a set of 3D voxel indices (x, y, z) representing hit voxels and
 * updates the corresponding voxels in the GPU-friendly voxel grid to occupied (value = 1).
 * 
 * @param unique_hit_voxels A set of 3D voxel indices (x, y, z) representing hit voxels.
 */
void Model::updateVoxelGridFromHits(const std::set<std::tuple<int, int, int>>& unique_hit_voxels) {
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
 * @brief Update the OctoMap based on precomputed octree hit keys.
 * 
 * This function takes a map of OctoMap keys representing hit voxels and
 * updates the corresponding voxels in the OctoMap to have a green color (0, 255, 0).
 * 
 * @param octree_hits A map where the key is the OctoMap voxel key (octomap::OcTreeKey),
 * and the value is a boolean indicating if the voxel was hit.
 */
void Model::updateOctomapWithHits(const std::unordered_map<octomap::OcTreeKey, bool, octomap::OcTreeKey::KeyHash>& octree_hits) {
    
    for (const auto& pair : octree_hits) {
        const octomap::OcTreeKey& key = pair.first;
        bool hit = pair.second;
        
        if (hit) {  // Only update for hit voxels (true values)
            // Update the voxel in the OctoMap and set its color to green (0, 255, 0)
            octomap::ColorOcTreeNode* node = surfaceShellOctomap_->updateNode(surfaceShellOctomap_->keyToCoord(key), true);
            if (node) {
                node->setColor(0, 255, 0);  // Green color for hit voxels
            } else {
                std::cerr << "Error: Failed to update OctoMap node!" << std::endl;
            }
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
bool Model::generateVoxelMap() {
    if (!surfaceShellOctomap_) {
        std::cerr << "Error: Surface shell OctoMap is not available." << std::endl;
        return false;
    }

    // Clear any existing data in the meta voxel map
    meta_voxel_map_.clear();

    // Iterate through each leaf node in the surface shell octomap
    for (auto it = surfaceShellOctomap_->begin_leafs(); it != surfaceShellOctomap_->end_leafs(); ++it) {
       
  
            octomap::OcTreeKey key = it.getKey();
            Eigen::Vector3d position(it.getX(), it.getY(), it.getZ());
            float occupancy = it->getOccupancy();
            MetaVoxel meta_voxel(position, key, occupancy);

            // Insert the MetaVoxel into the map
            meta_voxel_map_.setMetaVoxel(key, meta_voxel);
        
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
MetaVoxel* Model::getVoxel(const octomap::OcTreeKey& key) {
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
MetaVoxel* Model::getVoxel(const Eigen::Vector3d& position) {
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
bool Model::updateVoxelOccupancy(const octomap::OcTreeKey& key, float new_occupancy) {
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
bool Model::updateVoxelOccupancy(const Eigen::Vector3d& position, float new_occupancy) {
    MetaVoxel* voxel = getVoxel(position);
    if (voxel) {
        voxel->setOccupancy(new_occupancy);
        return true;
    }
    std::cerr << "MetaVoxel at the specified position not found." << std::endl;
    return false;
}


bool Model::addVoxelProperty(const std::string& property_name, const MetaVoxel::PropertyValue& initial_value) {
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
bool Model::setVoxelProperty(const octomap::OcTreeKey& key, const std::string& property_name, const MetaVoxel::PropertyValue& value) {
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
bool Model::setVoxelProperty(const Eigen::Vector3d& position, const std::string& property_name, const MetaVoxel::PropertyValue& value) {
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
MetaVoxel::PropertyValue Model::getVoxelProperty(const octomap::OcTreeKey& key, const std::string& property_name) const {
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
MetaVoxel::PropertyValue Model::getVoxelProperty(const Eigen::Vector3d& position, const std::string& property_name) const {
    octomap::OcTreeKey key;
    if (surfaceShellOctomap_ && surfaceShellOctomap_->coordToKeyChecked(octomap::point3d(position.x(), position.y(), position.z()), key)) {
        return meta_voxel_map_.getMetaVoxelProperty(key, property_name);
    }
    throw std::runtime_error("Failed to retrieve property for MetaVoxel at the specified position.");
}

void Model::clear() {
    // Mesh data
    meshData_.reset();
    minBound_ = octomap::point3d(0, 0, 0);
    maxBound_ = octomap::point3d(0, 0, 0);
    center_ = octomap::point3d(0, 0, 0);
    raycasting_scene_.reset();

    // Point cloud and voxel data
    pointCloud_.reset();
    volumetricPointCloud_.reset();
    pointCloudSpacing_ = 0.0;

    voxelGrid_.reset();
    if (gpu_voxel_grid_.voxel_data) {
        delete[] gpu_voxel_grid_.voxel_data;
        gpu_voxel_grid_.voxel_data = nullptr;
        gpu_voxel_grid_.width = gpu_voxel_grid_.height = gpu_voxel_grid_.depth = 0;
    }

    // Octree representations
    octoMap_.reset();
    volumetricOctomap_.reset();
    surfaceShellOctomap_.reset();
    explorationMap_.reset();
    octomap_resolution_ = 0.0;

    // Meta voxel map
    meta_voxel_map_.clear();
}


bool Model::compareVoxelStructures() const {
    bool consistent = true;

    std::cerr << "\nComparing Voxel Structures:\n";

    // Get the bounds of the surfaceShellOctomap_
    double min_x, min_y, min_z;
    surfaceShellOctomap_->getMetricMin(min_x, min_y, min_z);

    // Part 1: Check each voxel in surfaceShellOctomap_ against gpu_voxel_grid_ and meta_voxel_map_
    std::cerr << "Checking surfaceShellOctomap_ against gpu_voxel_grid_ and meta_voxel_map_\n";
    for (auto it = surfaceShellOctomap_->begin_leafs(); it != surfaceShellOctomap_->end_leafs(); ++it) {
        octomap::OcTreeKey key = it.getKey();
        Eigen::Vector3d voxel_center(it.getX(), it.getY(), it.getZ());

        // Check gpu_voxel_grid_
        int x_idx = static_cast<int>((voxel_center.x() - min_x) / voxel_size_);
        int y_idx = static_cast<int>((voxel_center.y() - min_y) / voxel_size_);
        int z_idx = static_cast<int>((voxel_center.z() - min_z) / voxel_size_);
        int linear_idx = z_idx * (gpu_voxel_grid_.width * gpu_voxel_grid_.height) + y_idx * gpu_voxel_grid_.width + x_idx;

        if (linear_idx < 0 || linear_idx >= gpu_voxel_grid_.width * gpu_voxel_grid_.height * gpu_voxel_grid_.depth || gpu_voxel_grid_.voxel_data[linear_idx] != 1) {
            std::cerr << "Mismatch in gpu_voxel_grid_ for voxel at (" << voxel_center.x() << ", " << voxel_center.y() << ", " << voxel_center.z() << ")\n";
            consistent = false;
        }

        // Check meta_voxel_map_
        MetaVoxel* meta_voxel = meta_voxel_map_.getMetaVoxel(key);
        if (!meta_voxel || meta_voxel->getPosition() != voxel_center) {
            std::cerr << "Mismatch in meta_voxel_map_ for voxel at (" << voxel_center.x() << ", " << voxel_center.y() << ", " << voxel_center.z() << ")\n";
            consistent = false;
        }
    }

    // Part 2: Check each voxel in gpu_voxel_grid_ against surfaceShellOctomap_ and meta_voxel_map_
    std::cerr << "Checking gpu_voxel_grid_ against surfaceShellOctomap_ and meta_voxel_map_\n";
    for (int z = 0; z < gpu_voxel_grid_.depth; ++z) {
        for (int y = 0; y < gpu_voxel_grid_.height; ++y) {
            for (int x = 0; x < gpu_voxel_grid_.width; ++x) {
                int linear_idx = z * (gpu_voxel_grid_.width * gpu_voxel_grid_.height) + y * gpu_voxel_grid_.width + x;
                if (gpu_voxel_grid_.voxel_data[linear_idx] == 1) {
                    // Convert (x, y, z) index to world coordinates
                    Eigen::Vector3d voxel_center(
                        min_x + x * voxel_size_ + voxel_size_ / 2,
                        min_y + y * voxel_size_ + voxel_size_ / 2,
                        min_z + z * voxel_size_ + voxel_size_ / 2
                    );

                    // Check surfaceShellOctomap_
                    octomap::OcTreeKey key = surfaceShellOctomap_->coordToKey(voxel_center.x(), voxel_center.y(), voxel_center.z());
                    if (!surfaceShellOctomap_->search(key)) {
                        std::cerr << "Voxel in gpu_voxel_grid_ does not exist in surfaceShellOctomap_ at (" << voxel_center.x() << ", " << voxel_center.y() << ", " << voxel_center.z() << ")\n";
                        consistent = false;
                    }

                    // Check meta_voxel_map_
                    MetaVoxel* meta_voxel = meta_voxel_map_.getMetaVoxel(key);
                    if (!meta_voxel || meta_voxel->getPosition() != voxel_center) {
                        std::cerr << "Voxel in gpu_voxel_grid_ does not exist in meta_voxel_map_ at (" << voxel_center.x() << ", " << voxel_center.y() << ", " << voxel_center.z() << ")\n";
                        consistent = false;
                    }
                }
            }
        }
    }

    // Part 3: Check each voxel in meta_voxel_map_ against surfaceShellOctomap_ and gpu_voxel_grid_
    std::cerr << "Checking meta_voxel_map_ against surfaceShellOctomap_ and gpu_voxel_grid_\n";
    for (const auto& entry : meta_voxel_map_.getMap()) {
        octomap::OcTreeKey key = entry.first;
        const MetaVoxel& meta_voxel = entry.second;
        Eigen::Vector3d voxel_center = meta_voxel.getPosition();

        // Check surfaceShellOctomap_
        if (!surfaceShellOctomap_->search(key)) {
            std::cerr << "Voxel in meta_voxel_map_ does not exist in surfaceShellOctomap_ at (" << voxel_center.x() << ", " << voxel_center.y() << ", " << voxel_center.z() << ")\n";
            consistent = false;
        }

        // Check gpu_voxel_grid_
        int x_idx = static_cast<int>((voxel_center.x() - min_x) / voxel_size_);
        int y_idx = static_cast<int>((voxel_center.y() - min_y) / voxel_size_);
        int z_idx = static_cast<int>((voxel_center.z() - min_z) / voxel_size_);
        int linear_idx = z_idx * (gpu_voxel_grid_.width * gpu_voxel_grid_.height) + y_idx * gpu_voxel_grid_.width + x_idx;

        if (linear_idx < 0 || linear_idx >= gpu_voxel_grid_.width * gpu_voxel_grid_.height * gpu_voxel_grid_.depth || gpu_voxel_grid_.voxel_data[linear_idx] != 1) {
            std::cerr << "Voxel in meta_voxel_map_ does not exist in gpu_voxel_grid_ at (" << voxel_center.x() << ", " << voxel_center.y() << ", " << voxel_center.z() << ")\n";
            consistent = false;
        }
    }

    if (consistent) {
        std::cerr << "All voxel structures are consistent.\n";
    } else {
        std::cerr << "Inconsistencies found in voxel structures.\n";
    }

    return consistent;
}

void Model::countAndCompareVoxelNumbers() const {
    // Count the number of leaf nodes (voxels) in surfaceShellOctomap_
    int surfaceShellVoxelCount = 0;
    for (auto it = surfaceShellOctomap_->begin_leafs(); it != surfaceShellOctomap_->end_leafs(); ++it) {
        surfaceShellVoxelCount++;
    }

    // Count the number of voxels in meta_voxel_map_
    int metaVoxelMapCount = meta_voxel_map_.size();

    // Print the counts
    std::cout << "Voxel Count Comparison:\n";
    std::cout << "  Surface Shell OctoMap Voxel Count: " << surfaceShellVoxelCount << std::endl;
    std::cout << "  Meta Voxel Map Voxel Count: " << metaVoxelMapCount << std::endl;

    // Check if they match
    if (surfaceShellVoxelCount == metaVoxelMapCount) {
        std::cout << "The voxel counts match." << std::endl;
    } else {
        std::cout << "Warning: The voxel counts do not match!" << std::endl;
    }
}


void Model::printMismatchedVoxels() const {
    std::cerr << "\nChecking for Mismatched Voxels between Surface Shell OctoMap and Meta Voxel Map:\n";
    bool mismatch_found = false;

    // Iterate through each voxel in surfaceShellOctomap_ and check against meta_voxel_map_
    for (auto it = surfaceShellOctomap_->begin_leafs(); it != surfaceShellOctomap_->end_leafs(); ++it) {
        octomap::OcTreeKey key = it.getKey();
        Eigen::Vector3d surfaceShellPosition(it.getX(), it.getY(), it.getZ());

        // Retrieve the corresponding MetaVoxel from meta_voxel_map_
        MetaVoxel* meta_voxel = meta_voxel_map_.getMetaVoxel(key);

        if (meta_voxel) {
            // Compare positions
            Eigen::Vector3d metaVoxelPosition = meta_voxel->getPosition();
            if (!metaVoxelPosition.isApprox(surfaceShellPosition)) {
                std::cerr << "Mismatch found for key (" << key.k[0] << ", " << key.k[1] << ", " << key.k[2] << "):\n";
                std::cerr << "  Surface Shell Position: (" << surfaceShellPosition.x() << ", " << surfaceShellPosition.y() << ", " << surfaceShellPosition.z() << ")\n";
                std::cerr << "  Meta Voxel Map Position: (" << metaVoxelPosition.x() << ", " << metaVoxelPosition.y() << ", " << metaVoxelPosition.z() << ")\n";
                mismatch_found = true;
            }
        } else {
            std::cerr << "Key present in Surface Shell OctoMap but missing in Meta Voxel Map: (" 
                      << key.k[0] << ", " << key.k[1] << ", " << key.k[2] << ")\n";
            mismatch_found = true;
        }
    }

    // Check if there are keys in meta_voxel_map_ that are not present in surfaceShellOctomap_
    for (const auto& entry : meta_voxel_map_.getMap()) {
        octomap::OcTreeKey key = entry.first;
        Eigen::Vector3d metaVoxelPosition = entry.second.getPosition();

        // Verify if the key exists in surfaceShellOctomap_
        if (!surfaceShellOctomap_->search(key)) {
            std::cerr << "Key present in Meta Voxel Map but missing in Surface Shell OctoMap: (" 
                      << key.k[0] << ", " << key.k[1] << ", " << key.k[2] << ")\n";
            std::cerr << "  Meta Voxel Map Position: (" << metaVoxelPosition.x() << ", " << metaVoxelPosition.y() << ", " << metaVoxelPosition.z() << ")\n";
            mismatch_found = true;
        }
    }

    if (!mismatch_found) {
        std::cerr << "No mismatches found. The voxel positions in Surface Shell OctoMap and Meta Voxel Map match.\n";
    }
}


} // namespace visioncraft


