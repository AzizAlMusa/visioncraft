import numpy as np
from visioncraft_py import MetaVoxel, MetaVoxelMap

# Initialize MetaVoxelMap and a sample MetaVoxel
meta_voxel_map = MetaVoxelMap()
sample_key = (100, 200, 300)
sample_position = np.array([1.0, 2.0, 3.0])
sample_voxel = MetaVoxel(sample_position, sample_key, 0.5)

def test_set_get_meta_voxel():
    print("Running test_set_get_meta_voxel...")
    
    # Insert MetaVoxel
    meta_voxel_map.setMetaVoxel(sample_key, sample_voxel)
    retrieved_voxel = meta_voxel_map.getMetaVoxel(sample_key)
    
    # Check if retrieved voxel matches the inserted one
    if retrieved_voxel:
        print("MetaVoxel position:", retrieved_voxel.getPosition())
        print("MetaVoxel occupancy:", retrieved_voxel.getOccupancy())
        assert np.allclose(retrieved_voxel.getPosition(), sample_position), "Position does not match"
        assert abs(retrieved_voxel.getOccupancy() - 0.5) < 1e-6, "Occupancy does not match"
    else:
        print("MetaVoxel not found")

def test_set_get_meta_voxel_property():
    print("Running test_set_get_meta_voxel_property...")

    # Insert MetaVoxel
    meta_voxel_map.setMetaVoxel(sample_key, sample_voxel)
    
    # Test setting and retrieving integer property
    meta_voxel_map.setMetaVoxelProperty(sample_key, "temperature", 25)
    temp = meta_voxel_map.getMetaVoxelProperty(sample_key, "temperature")
    print("Temperature property:", temp)
    assert temp == 25, "Temperature property does not match"
    
    # Test setting and retrieving float property
    meta_voxel_map.setMetaVoxelProperty(sample_key, "pressure", 101.3)
    pressure = meta_voxel_map.getMetaVoxelProperty(sample_key, "pressure")
    print("Pressure property:", pressure)
    assert abs(pressure - 101.3) < 1e-3, "Pressure property does not match"
    
    # Test setting and retrieving string property
    meta_voxel_map.setMetaVoxelProperty(sample_key, "status", "active")
    status = meta_voxel_map.getMetaVoxelProperty(sample_key, "status")
    print("Status property:", status)
    assert status == "active", "Status property does not match"
    
    # Test setting and retrieving vector property
    vector_property = np.array([0.5, 1.5, -0.5])
    meta_voxel_map.setMetaVoxelProperty(sample_key, "velocity", vector_property)
    velocity = meta_voxel_map.getMetaVoxelProperty(sample_key, "velocity")
    print("Velocity property:", velocity)
    assert np.allclose(velocity, vector_property), "Velocity property does not match"

def test_get_nonexistent_property():
    print("Running test_get_nonexistent_property...")
    
    # Attempt to retrieve a non-existent property
    try:
        meta_voxel_map.getMetaVoxelProperty(sample_key, "nonexistent_property")
    except RuntimeError as e:
        print("Caught expected error for non-existent property:", e)

def test_size_and_clear_map():
    print("Running test_size_and_clear_map...")
    
    # Insert MetaVoxels and check size
    meta_voxel_map.setMetaVoxel((100, 200, 300), sample_voxel)
    meta_voxel_map.setMetaVoxel((400, 500, 600), MetaVoxel(np.array([4.0, 5.0, 6.0]), (400, 500, 600), 0.7))
    
    print("MetaVoxelMap size:", meta_voxel_map.size())
    assert meta_voxel_map.size() == 2, "Map size does not match expected value"
    
    # Clear the map and verify
    meta_voxel_map.clear()
    print("MetaVoxelMap size after clear:", meta_voxel_map.size())
    assert meta_voxel_map.size() == 0, "Map should be empty after clear"

# Run the tests
test_set_get_meta_voxel()
test_set_get_meta_voxel_property()
test_get_nonexistent_property()
test_size_and_clear_map()

print("All tests completed.")
