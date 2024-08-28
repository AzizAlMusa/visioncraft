#ifndef VISIONCRAFT_UTILS_POISSON_SAMPLER_H
#define VISIONCRAFT_UTILS_POISSON_SAMPLER_H

#include <vector> ///< Include the vector header for STL containers.
#include <unordered_map> // Include unordered_map for storing triangle ID lists
#include <pcl/point_types.h> ///< Include PCL point types for 3D points.
#include <pcl/kdtree/kdtree_flann.h> ///< Include the KDTree header from PCL.
#include <vtkSmartPointer.h> ///< Include VTK smart pointers.
#include <vtkPolyData.h> ///< Include VTK PolyData for mesh representation.

namespace visioncraft {
namespace utils {

/**
 * @brief Struct to hold information about a point in PoissonSampler.
 */
struct PointInfo {
    pcl::PointXYZ point; ///< The 3D point.
    double weight;       ///< Weight of the point for sampling.
    bool deleted;        ///< Flag to indicate if the point is deleted.

    /**
     * @brief Construct a new PointInfo object.
     * 
     * @param p The 3D point.
     */
    PointInfo(const pcl::PointXYZ& p) : point(p), weight(0.0), deleted(false) {}
};

/**
 * @brief Class for generating point cloud using Poisson disk sampling.
 */
class PoissonSampler {
public:
    /**
     * @brief Default constructor for PoissonSampler.
     */
    PoissonSampler();

    // Function declarations
    double computeSurfaceArea(const vtkSmartPointer<vtkPolyData>& meshData); ///< Compute surface area of the mesh.
    void generateRandomPoint(const vtkSmartPointer<vtkPolyData>& meshData);
    int selectTriangleIndex(const std::vector<double>& areas); ///< Select a triangle index based on areas.
    void generateUniformSample(const vtkSmartPointer<vtkPolyData>& meshData, int numSamples);
    void generatePoissonDiskSample(const vtkSmartPointer<vtkPolyData>& meshData, int numSamples); ///< Generate point cloud using Poisson disk sampling.
    double calculateRMax(double surfaceArea, int numPoints); ///< Calculate the maximum radius for sampling.
    pcl::PointCloud<pcl::PointXYZ>::Ptr getPointCloud() const; ///< Get the generated point cloud.

private:
    std::vector<PointInfo> generatedPoints_; ///< Vector to store generated points.
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud_; ///< Pointer to the generated point cloud.
    std::vector<std::array<int, 3>> triangleIndexToVertexIDs_; // Map to store triangle ID lists 

};

} // namespace utils
} // namespace visioncraft

#endif // VISIONCRAFT_UTILS_POISSON_SAMPLER_H
