# tests/CMakeLists.txt


# Enable testing and add the test executables
enable_testing()
add_executable(meta_voxel_test cpp/meta_voxel_test.cpp)
# add_executable(meta_voxel_map_test cpp/meta_voxel_map_test.cpp)

# Link necessary libraries to the test executables
target_link_libraries(meta_voxel_test PRIVATE visioncraft_lib Open3D_no_OpenGL)
# target_link_libraries(meta_voxel_map_test PRIVATE visioncraft_lib)

# Register tests with CTest
add_test(NAME MetaVoxelTest COMMAND meta_voxel_test)
# add_test(NAME MetaVoxelMapTest COMMAND meta_voxel_map_test)
