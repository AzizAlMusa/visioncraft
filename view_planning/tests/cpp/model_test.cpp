// cpp/model_test.cpp

#include "visioncraft/model.h"
#include <gtest/gtest.h>
#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>
#include <open3d/Open3D.h>

using namespace visioncraft;

class ModelTest : public ::testing::Test {
protected:
    Model model;

    // Sample file path (Ensure this file exists for testing)
    std::string test_mesh_file = "../../models/cube.ply";

    void SetUp() override {
        // Make sure a valid mesh file is available for testing
        ASSERT_TRUE(model.loadMesh(test_mesh_file));
    }
};

TEST_F(ModelTest, LoadMesh) {
    // Verify that mesh is loaded correctly
    auto mesh = model.getMeshData();
    ASSERT_NE(mesh, nullptr) << "Mesh should be loaded and not null.";
    EXPECT_GT(mesh->vertices_.size(), 0) << "Loaded mesh should have vertices.";
}

TEST_F(ModelTest, InitializeRaycastingScene) {
    // Test initialization of the raycasting scene after loading mesh
    ASSERT_TRUE(model.initializeRaycastingScene()) << "Raycasting scene should initialize successfully.";
}

TEST_F(ModelTest, GeneratePointCloud) {
    // Initialize raycasting scene before generating point cloud
    ASSERT_TRUE(model.initializeRaycastingScene()) << "Raycasting scene must be initialized first.";
    ASSERT_TRUE(model.generatePointCloud(5000)) << "Point cloud generation should succeed.";

    auto pointCloud = model.getPointCloud();
    ASSERT_NE(pointCloud, nullptr) << "Generated point cloud should not be null.";
    EXPECT_GT(pointCloud->points_.size(), 0) << "Generated point cloud should contain points.";
}

TEST_F(ModelTest, GenerateOctoMap) {
    // Generate point cloud before octomap
    ASSERT_TRUE(model.initializeRaycastingScene()) << "Raycasting scene must be initialized first.";
    ASSERT_TRUE(model.generatePointCloud(5000)) << "Point cloud generation should succeed.";

    ASSERT_TRUE(model.generateOctoMap(0.1)) << "OctoMap generation should succeed with valid point cloud.";
    auto octomap = model.getOctomap();
    ASSERT_NE(octomap, nullptr) << "Generated OctoMap should not be null.";
    EXPECT_GT(octomap->size(), 0) << "OctoMap should contain nodes after generation.";
}

TEST_F(ModelTest, GenerateExplorationMap_ResolutionOnly) {
    // Generate exploration map with a given resolution
    octomap::point3d min_bound(-1.0, -1.0, -1.0);
    octomap::point3d max_bound(1.0, 1.0, 1.0);

    ASSERT_TRUE(model.generateExplorationMap(0.1, min_bound, max_bound)) << "Exploration map generation with resolution should succeed.";
    auto explorationMap = model.getExplorationMap();
    ASSERT_NE(explorationMap, nullptr) << "Generated exploration map should not be null.";
}

TEST_F(ModelTest, GenerateExplorationMap_NumCells) {
    // Ensure point cloud and raycasting are initialized first
    ASSERT_TRUE(model.initializeRaycastingScene()) << "Raycasting scene must be initialized first.";
    ASSERT_TRUE(model.generatePointCloud(5000)) << "Point cloud generation should succeed.";

    octomap::point3d min_bound(-1.0, -1.0, -1.0);
    octomap::point3d max_bound(1.0, 1.0, 1.0);
    ASSERT_TRUE(model.generateExplorationMap(32, min_bound, max_bound)) << "Exploration map generation with cell count should succeed.";
    
    auto explorationMap = model.getExplorationMap();
    ASSERT_NE(explorationMap, nullptr) << "Generated exploration map should not be null.";
    EXPECT_GT(explorationMap->size(), 0) << "Exploration map should contain nodes after generation.";
}

TEST_F(ModelTest, LoadModel) {
    // Test complete model loading and generation of all structures
    ASSERT_TRUE(model.loadModel(test_mesh_file, 50000)) << "Full model load should succeed.";

    auto mesh = model.getMeshData();
    ASSERT_NE(mesh, nullptr) << "Mesh data should not be null after loadModel.";

    auto pointCloud = model.getPointCloud();
    ASSERT_NE(pointCloud, nullptr) << "Point cloud should not be null after loadModel.";
    EXPECT_GT(pointCloud->points_.size(), 0) << "Point cloud should contain points after loadModel.";

    auto octomap = model.getOctomap();
    ASSERT_NE(octomap, nullptr) << "OctoMap should not be null after loadModel.";
    EXPECT_GT(octomap->size(), 0) << "OctoMap should contain nodes after loadModel.";
}


TEST_F(ModelTest, ConvertVoxelGridToGPUFormat) {
    // Prepare surfaceShellOctomap by loading a model and generating structures
    ASSERT_TRUE(model.loadModel(test_mesh_file, 50000));
    ASSERT_TRUE(model.generateSurfaceShellOctomap()) << "Surface shell OctoMap should be generated before converting voxel grid.";

    double voxelSize = 0.1;
    ASSERT_TRUE(model.convertVoxelGridToGPUFormat(voxelSize)) << "Voxel grid conversion to GPU format should succeed.";
    
    const auto& gpu_voxel_grid = model.getGPUVoxelGrid();
    EXPECT_EQ(gpu_voxel_grid.voxel_size, voxelSize) << "Voxel size should match the input parameter.";
    EXPECT_GT(gpu_voxel_grid.width, 0) << "Voxel grid width should be greater than zero.";
    EXPECT_GT(gpu_voxel_grid.height, 0) << "Voxel grid height should be greater than zero.";
    EXPECT_GT(gpu_voxel_grid.depth, 0) << "Voxel grid depth should be greater than zero.";
}

TEST_F(ModelTest, UpdateVoxelGridFromHits) {
    // Prepare surfaceShellOctomap and voxel grid
    ASSERT_TRUE(model.loadModel(test_mesh_file, 50000));
    ASSERT_TRUE(model.generateSurfaceShellOctomap());
    ASSERT_TRUE(model.convertVoxelGridToGPUFormat(0.1));

    // Define some hit voxels for updating
    std::set<std::tuple<int, int, int>> hit_voxels = {{1, 1, 1}, {2, 2, 2}};
    model.updateVoxelGridFromHits(hit_voxels);

    // Check that voxel grid reflects the updates
    const auto& gpu_voxel_grid = model.getGPUVoxelGrid();
    int idx_1_1_1 = 1 * (gpu_voxel_grid.width * gpu_voxel_grid.height) + 1 * gpu_voxel_grid.width + 1;
    int idx_2_2_2 = 2 * (gpu_voxel_grid.width * gpu_voxel_grid.height) + 2 * gpu_voxel_grid.width + 2;
    EXPECT_EQ(gpu_voxel_grid.voxel_data[idx_1_1_1], 1) << "Voxel at (1, 1, 1) should be marked as occupied.";
    EXPECT_EQ(gpu_voxel_grid.voxel_data[idx_2_2_2], 1) << "Voxel at (2, 2, 2) should be marked as occupied.";
}

TEST_F(ModelTest, UpdateOctomapWithHits) {
    // Ensure the surface shell OctoMap and GPU voxel grid are ready
    ASSERT_TRUE(model.loadModel(test_mesh_file, 50000));
    ASSERT_TRUE(model.generateSurfaceShellOctomap());
    ASSERT_TRUE(model.convertVoxelGridToGPUFormat(0.1));

    std::set<std::tuple<int, int, int>> hit_voxels = {{1, 1, 1}, {3, 3, 3}};
    model.updateOctomapWithHits(hit_voxels);

    // Validate OctoMap nodes at corresponding coordinates
    octomap::ColorOcTreeNode* node1 = model.getSurfaceShellOctomap()->search(0.1f + 1 * 0.1f, 0.1f + 1 * 0.1f, 0.1f + 1 * 0.1f);
    octomap::ColorOcTreeNode* node2 = model.getSurfaceShellOctomap()->search(0.1f + 3 * 0.1f, 0.1f + 3 * 0.1f, 0.1f + 3 * 0.1f);

    ASSERT_NE(node1, nullptr) << "Node at (1, 1, 1) should exist in OctoMap.";
    EXPECT_EQ(node1->getColor(), octomap::ColorOcTreeNode::Color(0, 255, 0)) << "Node at (1, 1, 1) should be colored green.";

    ASSERT_NE(node2, nullptr) << "Node at (3, 3, 3) should exist in OctoMap.";
    EXPECT_EQ(node2->getColor(), octomap::ColorOcTreeNode::Color(0, 255, 0)) << "Node at (3, 3, 3) should be colored green.";
}

TEST_F(ModelTest, GenerateVoxelMap) {
    // Generate surface shell OctoMap
    ASSERT_TRUE(model.loadModel(test_mesh_file, 50000));
    ASSERT_TRUE(model.generateSurfaceShellOctomap()) << "Surface shell OctoMap must be generated.";

    ASSERT_TRUE(model.generateVoxelMap()) << "Voxel map should generate based on surface shell OctoMap.";

    const MetaVoxelMap& voxel_map = model.getVoxelMap();
    EXPECT_GT(voxel_map.size(), 0) << "Generated voxel map should contain entries.";
}

TEST_F(ModelTest, GetVoxelByKey) {
    // Generate voxel map first
    ASSERT_TRUE(model.loadModel(test_mesh_file, 50000));
    ASSERT_TRUE(model.generateSurfaceShellOctomap());
    ASSERT_TRUE(model.generateVoxelMap());

    // Use a valid key from voxel map
    octomap::OcTreeKey key(1, 1, 1);  // Sample key (ensure this key exists in your test setup)
    MetaVoxel* voxel = model.getVoxel(key);
    ASSERT_NE(voxel, nullptr) << "Voxel should be retrievable by key.";
}

TEST_F(ModelTest, GetVoxelByPosition) {
    // Generate voxel map first
    ASSERT_TRUE(model.loadModel(test_mesh_file, 50000));
    ASSERT_TRUE(model.generateSurfaceShellOctomap());
    ASSERT_TRUE(model.generateVoxelMap());

    Eigen::Vector3d position(0.1, 0.1, 0.1);  // Adjust as necessary for your setup
    MetaVoxel* voxel = model.getVoxel(position);
    ASSERT_NE(voxel, nullptr) << "Voxel should be retrievable by position.";
}

TEST_F(ModelTest, UpdateVoxelOccupancyByKey) {
    ASSERT_TRUE(model.loadModel(test_mesh_file, 50000));
    ASSERT_TRUE(model.generateSurfaceShellOctomap());
    ASSERT_TRUE(model.generateVoxelMap());

    octomap::OcTreeKey key(1, 1, 1);
    ASSERT_TRUE(model.updateVoxelOccupancy(key, 0.8f)) << "Updating occupancy by key should succeed.";

    MetaVoxel* voxel = model.getVoxel(key);
    ASSERT_NE(voxel, nullptr) << "Voxel should be retrievable by key.";
    EXPECT_FLOAT_EQ(voxel->getOccupancy(), 0.8f) << "Voxel occupancy should match updated value.";
}

TEST_F(ModelTest, UpdateVoxelOccupancyByPosition) {
    ASSERT_TRUE(model.loadModel(test_mesh_file, 50000));
    ASSERT_TRUE(model.generateSurfaceShellOctomap());
    ASSERT_TRUE(model.generateVoxelMap());

    Eigen::Vector3d position(0.1, 0.1, 0.1);  // Adjust as necessary for your setup
    ASSERT_TRUE(model.updateVoxelOccupancy(position, 0.6f)) << "Updating occupancy by position should succeed.";

    MetaVoxel* voxel = model.getVoxel(position);
    ASSERT_NE(voxel, nullptr) << "Voxel should be retrievable by position.";
    EXPECT_FLOAT_EQ(voxel->getOccupancy(), 0.6f) << "Voxel occupancy should match updated value.";
}

TEST_F(ModelTest, AddVoxelProperty) {
    ASSERT_TRUE(model.loadModel(test_mesh_file, 50000));
    ASSERT_TRUE(model.generateSurfaceShellOctomap());
    ASSERT_TRUE(model.generateVoxelMap());

    std::string property_name = "temperature";
    MetaVoxel::PropertyValue initial_value = 25.0f;
    ASSERT_TRUE(model.addVoxelProperty(property_name, initial_value)) << "Adding voxel property should succeed.";
}

TEST_F(ModelTest, SetAndRetrieveVoxelPropertyByKey) {
    ASSERT_TRUE(model.loadModel(test_mesh_file, 50000));
    ASSERT_TRUE(model.generateSurfaceShellOctomap());
    ASSERT_TRUE(model.generateVoxelMap());

    octomap::OcTreeKey key(1, 1, 1);
    std::string property_name = "density";
    MetaVoxel::PropertyValue value = 5.0f;

    ASSERT_TRUE(model.setVoxelProperty(key, property_name, value)) << "Setting property by key should succeed.";
    auto retrieved_value = model.getVoxelProperty(key, property_name);
    EXPECT_EQ(boost::get<float>(retrieved_value), 5.0f) << "Retrieved value should match set value.";
}

TEST_F(ModelTest, SetAndRetrieveVoxelPropertyByPosition) {
    ASSERT_TRUE(model.loadModel(test_mesh_file, 50000));
    ASSERT_TRUE(model.generateSurfaceShellOctomap());
    ASSERT_TRUE(model.generateVoxelMap());

    Eigen::Vector3d position(0.1, 0.1, 0.1);
    std::string property_name = "pressure";
    MetaVoxel::PropertyValue value = 101.3f;

    ASSERT_TRUE(model.setVoxelProperty(position, property_name, value)) << "Setting property by position should succeed.";
    auto retrieved_value = model.getVoxelProperty(position, property_name);
    EXPECT_EQ(boost::get<float>(retrieved_value), 101.3f) << "Retrieved value should match set value.";
}