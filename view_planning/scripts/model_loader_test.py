import unittest
import visioncraft_py
import numpy as np

class TestModelLoader(unittest.TestCase):
    def setUp(self):
        # Initialize ModelLoader before each test
        self.loader = visioncraft_py.ModelLoader()
        self.mesh_path = "../models/cube.ply"

    def test_load_mesh(self):
        # Test loading a mesh
        success = self.loader.loadMesh(self.mesh_path)
        self.assertTrue(success, "Failed to load mesh.")
        mesh_data = self.loader.getMeshData()
        self.assertIsNotNone(mesh_data, "Mesh data should not be None after loading.")

    def test_initialize_raycasting_scene(self):
        # Load mesh before initializing the raycasting scene
        self.loader.loadMesh(self.mesh_path)
        success = self.loader.initializeRaycastingScene()
        self.assertTrue(success, "Failed to initialize raycasting scene.")

    def test_generate_point_cloud(self):
        # Load mesh and generate point cloud
        self.loader.loadMesh(self.mesh_path)
        success = self.loader.generatePointCloud(1000)
        self.assertTrue(success, "Failed to generate point cloud.")
        point_cloud = self.loader.getPointCloud()
        self.assertTrue(point_cloud.has_points(), "Generated point cloud should have points.")

    def test_generate_voxel_grid(self):
        # Load mesh, generate point cloud, and voxel grid
        self.loader.loadMesh(self.mesh_path)
        self.loader.generatePointCloud(1000)
        success = self.loader.generateVoxelGrid(0.05)
        self.assertTrue(success, "Failed to generate voxel grid.")
        voxel_grid = self.loader.getVoxelGrid()
        self.assertIsNotNone(voxel_grid, "Voxel grid should not be None.")
        self.assertTrue(len(voxel_grid.voxels) > 0, "Voxel grid should contain voxels.")

    def test_get_bounds(self):
        # Load mesh and check bounding box properties
        self.loader.loadMesh(self.mesh_path)
        min_bound = self.loader.getMinBound()
        max_bound = self.loader.getMaxBound()
        center = self.loader.getCenter()
        self.assertIsInstance(min_bound, np.ndarray, "Min bound should be an ndarray.")
        self.assertIsInstance(max_bound, np.ndarray, "Max bound should be an ndarray.")
        self.assertIsInstance(center, np.ndarray, "Center should be an ndarray.")

    def test_get_average_spacing(self):
        # Load mesh, generate point cloud, and test average spacing
        self.loader.loadMesh(self.mesh_path)
        self.loader.generatePointCloud(1000)
        spacing = self.loader.getAverageSpacing()
        self.assertGreater(spacing, 0, "Average spacing should be greater than 0.")

    def test_generate_octomap(self):
        # Load mesh, generate point cloud, and OctoMap
        self.loader.loadMesh(self.mesh_path)
        self.loader.generatePointCloud(1000)
        success = self.loader.generateOctoMap(0.05)
        self.assertTrue(success, "Failed to generate OctoMap.")
        octomap = self.loader.getOctomap()
        self.assertIsNotNone(octomap, "OctoMap should not be None.")

    def test_generate_surface_shell_octomap(self):
        # Load mesh, generate OctoMap, and generate surface shell OctoMap
        self.loader.loadMesh(self.mesh_path)
        self.loader.generatePointCloud(1000)
        self.loader.generateOctoMap(0.05)
        success = self.loader.generateSurfaceShellOctomap()
        self.assertTrue(success, "Failed to generate surface shell OctoMap.")
        surface_shell = self.loader.getSurfaceShellOctomap()
        self.assertIsNotNone(surface_shell, "Surface shell OctoMap should not be None.")

    def test_convert_voxel_grid_to_gpu_format(self):
        # Load mesh, generate voxel grid, and convert it to GPU format
        self.loader.loadMesh(self.mesh_path)
        self.loader.generatePointCloud(1000)
        self.loader.generateVoxelGrid(0.05)
        success = self.loader.convertVoxelGridToGPUFormat(0.05)
        self.assertTrue(success, "Failed to convert voxel grid to GPU format.")
        gpu_voxel_grid = self.loader.getGPUVoxelGrid()
        self.assertIsNotNone(gpu_voxel_grid, "GPU Voxel Grid should not be None.")

    def test_update_voxel_grid_from_hits(self):
        # Create a dummy hit voxel set and update GPU voxel grid
        hit_voxels = {(5, 5, 5), (10, 10, 10)}
        self.loader.updateVoxelGridFromHits(hit_voxels)
        gpu_voxel_grid = self.loader.getGPUVoxelGrid()
        # Check if updated voxel data reflects hits
        self.assertIn(1, gpu_voxel_grid.get_voxel_data(), "GPU voxel grid data should be updated with hits.")

    def test_update_octomap_with_hits(self):
        # Create a dummy hit voxel set and update OctoMap with hits
        self.loader.loadMesh(self.mesh_path)
        self.loader.generatePointCloud(1000)
        self.loader.generateOctoMap(0.05)
        hit_voxels = {(5, 5, 5), (10, 10, 10)}
        self.loader.updateOctomapWithHits(hit_voxels)
        octomap = self.loader.getOctomap()
        self.assertIsNotNone(octomap, "OctoMap should be available after update.")
        # Further checks can be added here to verify color changes in OctoMap

if __name__ == "__main__":
    unittest.main()
