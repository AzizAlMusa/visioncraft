import unittest
import sys
import os
import numpy as np
import pandas as pd

# Import the required classes
sys.path.append(os.path.abspath("../../build/python_bindings"))
from visioncraft_py import MetaVoxel, MetaVoxelMap

class TestMetaVoxelMap(unittest.TestCase):
    
    # Initialize a dictionary to track test results
    methods_status = {
        "Function Name": [],
        "Tested": [],
        "Status": []
    }
    
    def setUp(self):
        """Set up a new MetaVoxelMap and sample data for each test case."""
        self.meta_voxel_map = MetaVoxelMap()
        self.sample_key = (100, 200, 300)
        self.sample_position = np.array([1.0, 2.0, 3.0])
        self.sample_voxel = MetaVoxel(self.sample_position, self.sample_key, 0.5)
        self.meta_voxel_map.setMetaVoxel(self.sample_key, self.sample_voxel)

    def record_test(self, function_name, tested=True, status=True):
        """Record the test results for summary reporting."""
        self.methods_status["Function Name"].append(function_name)
        self.methods_status["Tested"].append("✔️" if tested else "❌")
        self.methods_status["Status"].append("✅" if status else "❌")
    
    def test_constructor(self):
        """Test the constructor of MetaVoxelMap."""
        print("Running test_constructor...")
        self.assertIsInstance(self.meta_voxel_map, MetaVoxelMap, "MetaVoxelMap not initialized correctly")
        self.record_test("MetaVoxelMap Constructor")

    def test_set_get_meta_voxel(self):
        """Test inserting and retrieving a MetaVoxel."""
        print("Running test_set_get_meta_voxel...")
        try:
            retrieved_voxel = self.meta_voxel_map.getMetaVoxel(self.sample_key)
            self.assertIsNotNone(retrieved_voxel, "MetaVoxel not found")
            np.testing.assert_allclose(retrieved_voxel.getPosition(), self.sample_position, err_msg="Position does not match")
            self.assertAlmostEqual(retrieved_voxel.getOccupancy(), 0.5, msg="Occupancy does not match")
            self.record_test("setMetaVoxel", status=True)
        except AssertionError:
            self.record_test("setMetaVoxel", status=False)

    def test_set_get_meta_voxel_property(self):
        """Test setting and retrieving various properties for a MetaVoxel."""
        print("Running test_set_get_meta_voxel_property...")
        try:
            # Integer property
            self.meta_voxel_map.setMetaVoxelProperty(self.sample_key, "temperature", 25)
            self.assertEqual(self.meta_voxel_map.getMetaVoxelProperty(self.sample_key, "temperature"), 25, "Temperature property does not match")
            
            # Float property
            self.meta_voxel_map.setMetaVoxelProperty(self.sample_key, "pressure", 101.3)
            self.assertAlmostEqual(self.meta_voxel_map.getMetaVoxelProperty(self.sample_key, "pressure"), 101.3, places=2, msg="Pressure property does not match")
            
            # String property
            self.meta_voxel_map.setMetaVoxelProperty(self.sample_key, "status", "active")
            self.assertEqual(self.meta_voxel_map.getMetaVoxelProperty(self.sample_key, "status"), "active", "Status property does not match")
            
            # Vector property
            vector_property = np.array([0.5, 1.5, -0.5])
            self.meta_voxel_map.setMetaVoxelProperty(self.sample_key, "velocity", vector_property)
            np.testing.assert_allclose(self.meta_voxel_map.getMetaVoxelProperty(self.sample_key, "velocity"), vector_property, err_msg="Velocity property does not match")
            
            self.record_test("setMetaVoxelProperty", status=True)
        except AssertionError:
            self.record_test("setMetaVoxelProperty", status=False)

    def test_get_nonexistent_property(self):
        """Test retrieving a non-existent property raises an error."""
        print("Running test_get_nonexistent_property...")
        try:
            with self.assertRaises(RuntimeError, msg="Expected error for non-existent property not raised"):
                self.meta_voxel_map.getMetaVoxelProperty(self.sample_key, "nonexistent_property")
            self.record_test("getMetaVoxelProperty", status=True)
        except AssertionError:
            self.record_test("getMetaVoxelProperty", status=False)

    def test_size(self):
        """Test the size function of the MetaVoxelMap."""
        print("Running test_size...")
        try:
            # Check initial size
            self.assertEqual(self.meta_voxel_map.size(), 1, "Map size does not match expected value after initial insert")
            self.record_test("size", status=True)
        except AssertionError:
            self.record_test("size", status=False)

    def test_clear_map(self):
        """Test the clear function of the MetaVoxelMap."""
        print("Running test_clear_map...")
        try:
            # Add another MetaVoxel
            new_key = (400, 500, 600)
            new_voxel = MetaVoxel(np.array([4.0, 5.0, 6.0]), new_key, 0.7)
            self.meta_voxel_map.setMetaVoxel(new_key, new_voxel)
            
            # Clear the map and check size
            self.meta_voxel_map.clear()
            self.assertEqual(self.meta_voxel_map.size(), 0, "Map should be empty after clear")
            self.record_test("clear", status=True)
        except AssertionError:
            self.record_test("clear", status=False)

    def test_set_property_for_all_voxels(self):
        """Test setting a property for all MetaVoxel instances in the map."""
        print("Running test_set_property_for_all_voxels...")
        try:
            # Add multiple voxels
            self.meta_voxel_map.setMetaVoxel((101, 201, 301), MetaVoxel(np.array([1.1, 2.1, 3.1]), (101, 201, 301), 0.6))
            self.meta_voxel_map.setMetaVoxel((102, 202, 302), MetaVoxel(np.array([1.2, 2.2, 3.2]), (102, 202, 302), 0.7))
            
            # Set the "initialized_property" for all voxels
            self.meta_voxel_map.setPropertyForAllVoxels("initialized_property", 1.0)
            
            # Check that each voxel has the "initialized_property" set
            for key in [(100, 200, 300), (101, 201, 301), (102, 202, 302)]:
                prop_value = self.meta_voxel_map.getMetaVoxelProperty(key, "initialized_property")
                self.assertEqual(prop_value, 1.0, f"initialized_property not set correctly for voxel at key {key}")
            
            self.record_test("setPropertyForAllVoxels", status=True)
        except AssertionError:
            self.record_test("setPropertyForAllVoxels", status=False)

    @classmethod
    def tearDownClass(cls):
        """Print the testing summary after all tests are run."""
        print("\nTesting Summary:")
        df = pd.DataFrame(cls.methods_status)
        print(df.to_markdown(index=False))

# Run the tests
if __name__ == '__main__':
    unittest.main()
