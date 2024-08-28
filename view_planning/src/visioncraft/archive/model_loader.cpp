#include "visioncraft/model_loader.h" // Include the ModelLoader header file
#include "visioncraft/utils/poisson_sampler.h" // Include the header file for PoissonSampler

#include <iostream>
#include <string>

#include <vtkSmartPointer.h>
#include <vtkSTLReader.h>
#include <vtkPLYReader.h>

namespace visioncraft {

// Constructor for the ModelLoader class
ModelLoader::ModelLoader() :  pointCloud_(new pcl::PointCloud<pcl::PointNormal>) {
    std::cout << "ModelLoader constructor called." << std::endl;
}

// Destructor for the ModelLoader class
ModelLoader::~ModelLoader() {
    std::cout << "ModelLoader destructor called." << std::endl;
}

// Load a mesh from a file
bool ModelLoader::loadMesh(const std::string& file_path) {
    // Check if the file has the correct extension
    std::string extension = file_path.substr(file_path.find_last_of(".") + 1);
    if (extension == "stl" || extension == "ply") {
        // Read the mesh file using VTK based on the file extension
        vtkSmartPointer<vtkPolyData> polyData = nullptr;
        if (extension == "stl") {
            vtkSmartPointer<vtkSTLReader> stlReader = vtkSmartPointer<vtkSTLReader>::New();
            stlReader->SetFileName(file_path.c_str());
            stlReader->Update();
            polyData = stlReader->GetOutput();
        } else if (extension == "ply") {
            vtkSmartPointer<vtkPLYReader> plyReader = vtkSmartPointer<vtkPLYReader>::New();
            plyReader->SetFileName(file_path.c_str());
            plyReader->Update();
            polyData = plyReader->GetOutput();
        }

        // Check if the polydata is valid
        if (polyData != nullptr) {
            meshData_ = polyData;
            // Return true if the mesh data is successfully loaded
            return true;
        } else {
            std::cerr << "Error: Failed to load mesh data from file." << std::endl;
        }
    } else {
        std::cerr << "Error: Unsupported file extension. Only STL and PLY files are supported." << std::endl;
    }
    // Return false if the file extension is not supported or loading fails
    return false;
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
    std::cout << "Number of Points: " << meshData_->GetNumberOfPoints() << std::endl;
    std::cout << "Number of Cells: " << meshData_->GetNumberOfCells() << std::endl;

    // Create a PoissonSampler object to generate the point cloud
    utils::PoissonSampler poissonSampler;
    poissonSampler.generatePoissonDiskSample(meshData_, numSamples);

    pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud(new pcl::PointCloud<pcl::PointXYZ>);
    tempCloud = poissonSampler.getPointCloud();
    double spacing = calculateAverageSpacing<pcl::PointCloud<pcl::PointXYZ>>(tempCloud);

    // Create normal estimator
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimator;
    normalEstimator.setInputCloud(tempCloud);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    normalEstimator.setSearchMethod(tree);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    normalEstimator.setRadiusSearch(5 * spacing);  // Set the radius of the normal estimation
    normalEstimator.compute(*normals);

    // Combine points and normals
    pcl::concatenateFields(*tempCloud, *normals, *pointCloud_);

    // Print the number of points generated using Poisson disk sampling
    std::cout << "Generated " << pointCloud_->size() << " points using Poisson disk sampling." << std::endl;

     // Print 10 normals values
    std::cout << "First 10 normals values:" << std::endl;
    for (int i = 0; i < 10 && i < normals->size(); ++i) {
        std::cout << "Normal " << i << ": " << normals->points[i].normal_x << ", "
                  << normals->points[i].normal_y << ", " << normals->points[i].normal_z << std::endl;
    }

    return true;
}

template<typename PointCloudT>
double ModelLoader::calculateAverageSpacing(const typename PointCloudT::Ptr& pointCloud) {
    if (!pointCloud || pointCloud->empty()) {
        std::cerr << "Error: Invalid or empty point cloud." << std::endl;
        return 0.0;  // Return 0.0 if the point cloud is invalid or empty
    }

    // Create KD-Tree for efficient nearest neighbor search
    pcl::KdTreeFLANN<typename PointCloudT::PointType> kdtree;
    kdtree.setInputCloud(pointCloud);

    double totalSpacing = 0.0;
    int numPoints = 0;

    // Iterate through each point in the point cloud
    for (const auto& point : pointCloud->points) {
        std::vector<int> indices(1);
        std::vector<float> squaredDistances(1);

        // Search for the nearest neighbor
        if (kdtree.nearestKSearch(point, 2, indices, squaredDistances) > 0) {
            totalSpacing += std::sqrt(squaredDistances[1]);  // Use the squared distance of the 2nd neighbor
            numPoints++;
        }
    }

    // Calculate the average spacing
    double averageSpacing = (numPoints > 0) ? (totalSpacing / numPoints) : 0.0;
    return averageSpacing;
}



} // namespace visioncraft
