#ifndef VISIONCRAFT_MODEL_LOADER_H
#define VISIONCRAFT_MODEL_LOADER_H

#include <string>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>

#include <vtkSmartPointer.h>
#include <vtkPolyData.h>

namespace visioncraft {

/**
 * @brief Class for loading and processing 3D models.
 * 
 * This class provides functionality to load 3D models in various formats
 * using VTK and process them for various applications.
 */
class ModelLoader {
public:
    /**
     * @brief Default constructor for ModelLoader class.
     */
    ModelLoader();

    /**
     * @brief Destructor for ModelLoader class.
     */
    ~ModelLoader();

    /**
     * @brief Load a 3D mesh from a file.
     * 
     * This function loads a 3D mesh from a file in STL or PLY format using VTK.
     * 
     * @param file_path The path to the mesh file.
     * @return True if the mesh is loaded successfully, false otherwise.
     */
    bool loadMesh(const std::string& file_path);

    /**
     * @brief Generate a point cloud from the loaded mesh using Poisson disk sampling.
     * 
     * This function generates a point cloud from the loaded mesh using Poisson disk sampling.
     * 
     * @param numSamples The number of samples to use for Poisson disk sampling.
     * @return True if the point cloud is generated successfully, false otherwise.
     */
    bool generatePointCloud(int numSamples);

    /**
     * @brief Get the mesh data.
     * 
     * @return A smart pointer to the vtkPolyData representing the mesh.
     */
    vtkSmartPointer<vtkPolyData> getMeshData() const { return meshData_; }

    /**
     * @brief Get the point cloud data.
     * 
     * @return A pointer to the pcl::PointCloud representing the point cloud.
     */
    pcl::PointCloud<pcl::PointNormal>::Ptr getPointCloud() const { return pointCloud_; }

    /**
     * @brief Calculate the average spacing of the point cloud points and set the private variable pointCloudSpacing_.
     * 
     * This function calculates the average spacing of the points in the point cloud
     * by finding the nearest neighbor for each point and averaging the distances.
     * The calculated average spacing is then set to the private variable pointCloudSpacing_.
     */
     template<typename PointCloudT>
     double calculateAverageSpacing(const typename PointCloudT::Ptr& pointCloud);

    /**
     * @brief Get the average spacing of the point cloud points.
     * 
     * @return The average spacing of the point cloud points.
     */
    double getAverageSpacing() const { return pointCloudSpacing_; }

    void correctNormalsUsingSignedDistance(double epsilon);

private:
    vtkSmartPointer<vtkPolyData> meshData_;  ///< Mesh data representing the loaded 3D model.
    pcl::PointCloud<pcl::PointNormal>::Ptr pointCloud_;  ///< Point cloud data extracted from the mesh.
    double pointCloudSpacing_;  ///< Spacing between points in the generated point cloud.

};

} // namespace visioncraft

#endif // VISIONCRAFT_MODEL_LOADER_H
