#ifndef VISIONCRAFT_MODEL_LOADER_H
#define VISIONCRAFT_MODEL_LOADER_H

#include <string>
#include <memory>
#include <open3d/Open3D.h>
#include <octomap/ColorOcTree.h>
#include "open3d/t/geometry/RaycastingScene.h"

namespace visioncraft {




/**
 * @brief Struct representing a GPU-friendly voxel grid.
 * 
 * This struct stores the voxel grid as a linearized 1D array for efficient access on the GPU.
 * It includes the dimensions of the grid (width, height, depth) and the voxel size.
 */
struct VoxelGridGPU {
    int* voxel_data;  ///< Linearized array for voxel data (0 = free, 1 = occupied).
    int width, height, depth;  ///< Dimensions of the voxel grid (number of voxels along each axis).
    double voxel_size;  ///< Size of each voxel.
    float min_bound[3];  ///< Minimum bound of the voxel grid.

    /**
     * @brief Default constructor initializing the voxel grid with zero values.
     */
    VoxelGridGPU() : voxel_data(nullptr), width(0), height(0), depth(0), voxel_size(0.0) {}
};    

/**
 * @brief Class for loading and processing 3D models using Open3D.
 * 
 * This class provides functionality to load 3D models in various formats
 * using Open3D/Octomap and process them for various applications. It replaces previous
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
     * This function loads a 3D mesh from a file in STL, OBJ, PLY, or other supported formats
     * using Open3D's mesh reading capabilities.
     * 
     * @param file_path The path to the mesh file.
     * @return True if the mesh is loaded successfully, false otherwise.
     */
    bool loadMesh(const std::string& file_path);

    /**
     * @brief Load a 3D model and generate all necessary structures.
     * 
     * This function loads a 3D model and generates the mesh, point cloud, voxel grid, and octomap.
     * 
     * @param file_path The path to the mesh file.
     * @param num_samples The number of samples for point cloud generation.
     * @param resolution The resolution for voxel grid and octomap generation.
     * @return True if the model is loaded and all structures are generated successfully, false otherwise.
     */
    bool loadModel(const std::string& file_path, int num_samples = 10000, double resolution = -1);

    /**
     * @brief Load a 3D model and generate all necessary structures for exploration.
     * 
     * This function loads a 3D model and generates the mesh, point cloud, exploration map, voxel grid, and octomap.
     * This is intended when the model is used for exploration and the exploration map is needed.
     * 
     * @param file_path The path to the mesh file.
     * @param num_samples The number of samples for point cloud generation.
     * @param num_cells_per_side The number of cells per side for the exploration octree.
     * @return True if the model is loaded and all structures are generated successfully, false otherwise.
     */
    bool loadExplorationModel(const std::string& file_path, int num_samples = 10000, int num_cells_per_side = 32);


     /**
     * @brief Initialize the open3d raycasting scene with the current mesh.
     * 
     * This function initializes the raycasting scene using the loaded mesh.
     * 
     * @return True if the raycasting scene is initialized successfully, false otherwise.
     */
    bool initializeRaycastingScene();

    /**
     * @brief Generate a point cloud from the loaded mesh using Poisson disk sampling.
     * 
     * This function generates a point cloud from the loaded mesh using Poisson disk sampling,
     * leveraging Open3D's point cloud sampling functionalities.
     * 
     * @param numSamples The number of samples to use for Poisson disk sampling.
     * @return True if the point cloud is generated successfully, false otherwise.
     */
    bool generatePointCloud(int numSamples);

    /**
     * @brief Fill the volumetric point cloud with uniformly spaced points based on the average spacing.
     * 
     * This function generates a uniformly spaced point cloud within the bounding box of the mesh.
     * 
     * @return True if the volumetric point cloud is generated successfully, false otherwise.
     */
    bool generateVolumetricPointCloud();

     /**
     * @brief Get the volumetric point cloud data.
     * 
     * @return A shared pointer to the volumetric Open3D PointCloud.
     */
    std::shared_ptr<open3d::geometry::PointCloud> getVolumetricPointCloud() const { return volumetricPointCloud_; }


    /**
     * @brief Get the mesh data.
     * 
     * @return A shared pointer to the Open3D TriangleMesh representing the mesh.
     */
    std::shared_ptr<open3d::geometry::TriangleMesh> getMeshData() const { return meshData_; }

     /**
     * @brief Get the minimum bound of the mesh bounding box.
     * 
     * @return The minimum bound as an octomap::point3d.
     */
    octomap::point3d getMinBound() const { return minBound_; }

    /**
     * @brief Get the maximum bound of the mesh bounding box.
     * 
     * @return The maximum bound as an octomap::point3d.
     */
    octomap::point3d getMaxBound() const { return maxBound_; }

    /**
     * @brief Get the center of the mesh bounding box.
     * 
     * @return The center as an octomap::point3d.
     */
    octomap::point3d getCenter() const { return center_; }



    /**
     * @brief Get the point cloud data.
     * 
     * @return A shared pointer to the Open3D PointCloud representing the point cloud.
     */
    std::shared_ptr<open3d::geometry::PointCloud> getPointCloud() const { return pointCloud_; }

 
    /**
     * @brief Get the average spacing of the point cloud points.
     * 
     * @return The average spacing of the point cloud points.
     */
    double getAverageSpacing() const { return pointCloudSpacing_; }

    /**
     * @brief Correct normals in the point cloud using a signed distance method with an epsilon threshold.
     * 
     * This function corrects the normals of points in the point cloud that are close to the mesh surface,
     * ensuring the normals are consistently oriented based on the signed distance computation.
     * 
     * @param epsilon A threshold value to determine how close points must be to the surface to adjust normals.
     */
    std::shared_ptr<open3d::geometry::PointCloud> correctNormalsUsingSignedDistance(std::shared_ptr<open3d::geometry::PointCloud> pointcloud, double epsilon);

    /**
     * @brief Generate a voxel grid representation of the object.
     * 
     * This function generates a voxel grid representation of the object using Open3D's VoxelGrid
     * functionalities. The voxel grid can be used for various applications such as collision checking
     * and occupancy mapping.
     * 
     * @param voxelSize The size of the voxels in the grid.
     * @return True if the voxel grid is generated successfully, false otherwise.
     */
    bool generateVoxelGrid(double voxelSize);

    /**
     * @brief Get the voxel grid data.
     * 
     * @return A shared pointer to the Open3D VoxelGrid representing the voxel grid.
     */
    std::shared_ptr<open3d::geometry::VoxelGrid> getVoxelGrid() const { return voxelGrid_; }


     /**
     * @brief Generate an OctoMap representation of the object.
     * 
     * This function generates an OctoMap representation of the object using the point cloud data.
     * 
     * @param resolution The resolution of the OctoMap.
     * @return True if the OctoMap is generated successfully, false otherwise.
     */
    bool generateOctoMap(double resolution);

    /**
     * @brief Get the colored OctoMap data.
     * 
     * @return A shared pointer to the colored OctoMap representing the object.
     */
    std::shared_ptr<octomap::ColorOcTree> getOctomap() const { return octoMap_; }

    /**
     * @brief Generate an OctoMap representation of the volumetric point cloud.
     * 
     * This function generates an OctoMap representation using the volumetric point cloud data.
     * 
     * @param resolution The resolution of the OctoMap.
     * @return True if the OctoMap is generated successfully, false otherwise.
     */
    bool generateVolumetricOctoMap(double resolution);

    /**
     * @brief Get the volumetric OctoMap data.
     * 
     * @return A shared pointer to the volumetric OctoMap.
     */
    std::shared_ptr<octomap::ColorOcTree> getVolumetricOctomap() const { return volumetricOctomap_; }


    /**
     * @brief Generate the surface shell OctoMap by eroding the volumetric OctoMap.
     * 
     * This function generates the surface shell OctoMap by performing erosion on the volumetric OctoMap
     * with a structuring element that has one center element and 6 neighbors along the principal axes.
     * 
     * @return True if the surface shell OctoMap is generated successfully, false otherwise.
     */
    bool generateSurfaceShellOctomap();

    /**
     * @brief Get the surface shell OctoMap data.
     * 
     * @return A shared pointer to the surface shell OctoMap.
     */
    std::shared_ptr<octomap::ColorOcTree> getSurfaceShellOctomap() const { return surfaceShellOctomap_; }



     /**
     * @brief Generate an exploration map with a given resolution.
     * 
     * This function generates an exploration map, which is a dense map with occupancy values of 0.5
     * within the specified bounding box.
     * 
     * @param resolution The resolution of the exploration map.
     * @param min_bound The minimum coordinates of the bounding box.
     * @param max_bound The maximum coordinates of the bounding box.
     * @return True if the exploration map is generated successfully, false otherwise.
     */
    bool generateExplorationMap(double resolution, const octomap::point3d& min_bound, const octomap::point3d& max_bound);

    /**
     * @brief Overloaded function to generate an exploration map with a specified number of cells per side.
     * 
     * This function generates an exploration map in the form of a cube with the specified number of cells per side.
     * 
     * @param num_cells_per_side The number of cells per side of the cube.
     * @param min_bound The minimum coordinates of the bounding box.
     * @param max_bound The maximum coordinates of the bounding box.
     * @return True if the exploration map is generated successfully, false otherwise.
     */
    bool generateExplorationMap(int num_cells_per_side, const octomap::point3d& min_bound, const octomap::point3d& max_bound);


    /**
     * @brief Get the exploration map data.
     * 
     * @return A shared pointer to the exploration map.
     */
    std::shared_ptr<octomap::ColorOcTree> getExplorationMap() const { return explorationMap_; }


    /**
     * @brief Get the computed resolution for the exploration octomap based on num_cells_per_side.
     * 
     * @return resolution of the exploration map.
     */
    double getExplorationMapResolution() const { return octomap_resolution_; }
    

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
    bool convertVoxelGridToGPUFormat(double voxelSize);

    /**
     * @brief Get the GPU-friendly voxel grid data.
     * 
     * This function provides access to the voxel grid data that has been converted
     * for use on the GPU.
     * 
     * @return A reference to the VoxelGridGPU structure containing the voxel grid data.
     */
    const VoxelGridGPU& getGPUVoxelGrid() const { return gpu_voxel_grid_; }

    /**
     * @brief Update the voxel grid based on the hit voxels.
     * 
     * This function takes a set of 3D voxel indices (x, y, z) representing hit voxels and
     * updates the corresponding voxels in the GPU-friendly voxel grid to occupied (value = 1).
     * 
     * @param unique_hit_voxels A set of 3D voxel indices (x, y, z) representing hit voxels.
     */
    void updateVoxelGridFromHits(const std::set<std::tuple<int, int, int>>& unique_hit_voxels);

    /**
     * @brief Update the OctoMap based on the hit voxels.
     * 
     * This function takes a set of 3D voxel indices (x, y, z), converts them to world coordinates,
     * and updates the corresponding voxels in the OctoMap to have a green color (0, 255, 0).
     * 
     * @param unique_hit_voxels A set of 3D voxel indices (x, y, z) representing hit voxels.
     */
    void updateOctomapWithHits(const std::set<std::tuple<int, int, int>>& unique_hit_voxels);


private:

    /**
     * @brief Generate all necessary structures from the loaded mesh.
     * 
     * This function generates the point cloud, volumetric point cloud, voxel grid, and octomap
     * based on the loaded mesh. It uses the specified number of samples and resolution, or defaults
     * if these parameters are not provided.
     * 
     * @param num_samples The number of samples for point cloud generation.
     * @param resolution The resolution for voxel grid and octomap generation.
     * @return True if all structures are generated successfully, false otherwise.
     */
    bool generateAllStructures(int num_samples, double resolution);


    /**
     * @brief Generate all necessary structures from the loaded mesh.
     * 
     * This function generates the point cloud, volumetric point cloud, exploration map, octomap, surface octomap and 
     * voxel grid based on the loaded mesh. It uses the specified number of samples and number of cells per side for the exploration map
     * , or defaults if these parameters are not provided.
     * 
     * @param num_samples The number of samples for point cloud generation.
     * @param num_cells_per_side The number of cells per side for exploration map generation.
     * @return True if all structures are generated successfully, false otherwise.
     */
    bool generateExplorationStructures(int num_samples, int num_cells_per_side);

    // Mesh representation
    std::shared_ptr<open3d::geometry::TriangleMesh> meshData_;  ///< Mesh data representing the loaded 3D model.
    octomap::point3d minBound_;  ///< Minimum bound of the mesh bounding box.
    octomap::point3d maxBound_;  ///< Maximum bound of the mesh bounding box.
    octomap::point3d center_;    ///< Center of the mesh bounding box.
    std::shared_ptr<open3d::t::geometry::RaycastingScene> raycasting_scene_; ///< Raycasting scene for various computations


    // Point cloud representation
    std::shared_ptr<open3d::geometry::PointCloud> pointCloud_;  ///< Point cloud data extracted from the mesh.
    double pointCloudSpacing_;  ///< Spacing between points in the generated point cloud.
    std::shared_ptr<open3d::geometry::PointCloud> volumetricPointCloud_; ///< Volumetric point cloud with uniformly spaced points


    // Voxel representation
    std::shared_ptr<open3d::geometry::VoxelGrid> voxelGrid_;  ///< Voxel grid representation of the object.

    VoxelGridGPU gpu_voxel_grid_; ///< GPU-friendly voxel grid for raycasting operations.


    // Octree representation
    std::shared_ptr<octomap::ColorOcTree> octoMap_; ///< OctoMap representation of the object.
    std::shared_ptr<octomap::ColorOcTree> volumetricOctomap_; ///< OctoMap representation based on the volumetric point cloud
    std::shared_ptr<octomap::ColorOcTree> surfaceShellOctomap_; ///< OctoMap representation of the surface shell.
    std::shared_ptr<octomap::ColorOcTree> explorationMap_; ///< OctoMap representation of the regions that cells that were scanned.
    double octomap_resolution_; ///< Resolution of the OctoMap used to match the explorationMap that will be invoked by a number of cells per side rather than resolution.

    // Error and Uncertainty Margins
    // TODO: Add MMC and LMC octomaps


    // Features and model properties

    // TODO: Add FPFH for the octomap/voxels
    // TODO: Add normals for the octomap/voxels
    // TODO: Add FPFH clustering for the octomap/voxels
    // TODO: Add features such as NARF etc. for the octomap/voxels
};  

} // namespace visioncraft

#endif // VISIONCRAFT_MODEL_LOADER_H
