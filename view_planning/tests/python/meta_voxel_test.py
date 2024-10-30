import unittest
import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath("../../build/python_bindings"))
from visioncraft_py import MetaVoxel

class TestMetaVoxel(unittest.TestCase):
    
    # Initialize a dictionary to track test results
    methods_status = {
        "Function Name": [],
        "Tested": [],
        "Status": []
    }
    
    def setUp(self):
        """Set up a new MetaVoxel instance for each test case."""
        self.sample_position = np.array([1.0, 2.0, 3.0])
        self.sample_key = (100, 200, 300)
        self.sample_occupancy = 0.5
        self.meta_voxel = MetaVoxel(self.sample_position, self.sample_key, self.sample_occupancy)

    def record_test(self, function_name, tested=True, status=True):
        """Record the test results for summary reporting."""
        self.methods_status["Function Name"].append(function_name)
        self.methods_status["Tested"].append("✔️" if tested else "❌")
        self.methods_status["Status"].append("✅" if status else "❌")
    
    def test_constructor(self):
        """Test the constructor of MetaVoxel."""
        print("Running test_constructor...")
        try:
            # Check if the MetaVoxel instance was initialized correctly
            np.testing.assert_allclose(self.meta_voxel.getPosition(), self.sample_position, err_msg="Position does not match")
            self.assertAlmostEqual(self.meta_voxel.getOccupancy(), self.sample_occupancy, msg="Occupancy does not match")
            self.assertEqual(self.meta_voxel.getOctomapKey(), self.sample_key, "Octomap key does not match")
            self.record_test("MetaVoxel Constructor", status=True)
        except AssertionError:
            self.record_test("MetaVoxel Constructor", status=False)

    def test_set_get_position(self):
        """Test getting position of MetaVoxel."""
        print("Running test_set_get_position...")
        try:
            np.testing.assert_allclose(self.meta_voxel.getPosition(), self.sample_position, err_msg="Position does not match")
            self.record_test("getPosition", status=True)
        except AssertionError:
            self.record_test("getPosition", status=False)

    def test_set_get_occupancy(self):
        """Test setting and retrieving occupancy."""
        print("Running test_set_get_occupancy...")
        try:
            self.meta_voxel.setOccupancy(0.8)
            self.assertAlmostEqual(self.meta_voxel.getOccupancy(), 0.8, msg="Occupancy does not match after setOccupancy")
            self.record_test("setOccupancy/getOccupancy", status=True)
        except AssertionError:
            self.record_test("setOccupancy/getOccupancy", status=False)

    def test_set_get_log_odds(self):
        """Test setting and retrieving log-odds."""
        print("Running test_set_get_log_odds...")
        try:
            log_odds_value = -0.2
            self.meta_voxel.setLogOdds(log_odds_value)
            self.assertAlmostEqual(self.meta_voxel.getLogOdds(), log_odds_value, msg="Log-odds does not match after setLogOdds")
            expected_occupancy = 1.0 / (1.0 + np.exp(-log_odds_value))
            self.assertAlmostEqual(self.meta_voxel.getOccupancy(), expected_occupancy, places=2, msg="Occupancy does not match expected value after setting log-odds")
            self.record_test("setLogOdds/getLogOdds", status=True)
        except AssertionError:
            self.record_test("setLogOdds/getLogOdds", status=False)

    def test_set_get_property(self):
        """Test setting and retrieving custom properties."""
        print("Running test_set_get_property...")
        try:
            # Integer property
            self.meta_voxel.setProperty("temperature", 25)
            self.assertEqual(self.meta_voxel.getProperty("temperature"), 25, "Temperature property does not match")
            
            # Float property
            self.meta_voxel.setProperty("pressure", 101.3)
            self.assertAlmostEqual(self.meta_voxel.getProperty("pressure"), 101.3, places=2, msg="Pressure property does not match")
            
            # String property
            self.meta_voxel.setProperty("status", "active")
            self.assertEqual(self.meta_voxel.getProperty("status"), "active", "Status property does not match")
            
            # Vector property
            vector_property = np.array([0.5, 1.5, -0.5])
            self.meta_voxel.setProperty("velocity", vector_property)
            np.testing.assert_allclose(self.meta_voxel.getProperty("velocity"), vector_property, err_msg="Velocity property does not match")
            
            self.record_test("setProperty/getProperty", status=True)
        except AssertionError:
            self.record_test("setProperty/getProperty", status=False)

    def test_has_property(self):
        """Test checking existence of a property."""
        print("Running test_has_property...")
        try:
            self.meta_voxel.setProperty("temperature", 25)
            self.assertTrue(self.meta_voxel.hasProperty("temperature"), "hasProperty failed for existing property")
            self.assertFalse(self.meta_voxel.hasProperty("nonexistent_property"), "hasProperty should return False for non-existent property")
            self.record_test("hasProperty", status=True)
        except AssertionError:
            self.record_test("hasProperty", status=False)

    def test_get_nonexistent_property(self):
        """Test retrieving a non-existent property raises an error."""
        print("Running test_get_nonexistent_property...")
        try:
            with self.assertRaises(RuntimeError, msg="Expected error for non-existent property not raised"):
                self.meta_voxel.getProperty("nonexistent_property")
            self.record_test("getProperty", status=True)
        except AssertionError:
            self.record_test("getProperty", status=False)

    @classmethod
    def tearDownClass(cls):
        """Print the testing summary after all tests are run."""
        print("\nTesting Summary:")
        df = pd.DataFrame(cls.methods_status)
        print(df.to_markdown(index=False))

# Run the tests
if __name__ == '__main__':
    unittest.main()
