# tests/CMakeLists.txt

# Enable testing
enable_testing()

# Find Google Test
# Find Google Test
find_package(GTest REQUIRED)
if(NOT GTEST_FOUND)
    message(FATAL_ERROR "GTest not found.")
endif()

include_directories(${GTEST_INCLUDE_DIRS})

# Add test executables
add_executable(meta_voxel_test cpp/meta_voxel_test.cpp)
add_executable(meta_voxel_map_test cpp/meta_voxel_map_test.cpp)
add_executable(model_test cpp/model_test.cpp)

# Link test executables with necessary libraries
target_link_libraries(meta_voxel_test PRIVATE visioncraft_lib project_dependencies GTest::GTest GTest::Main)
target_link_libraries(meta_voxel_map_test PRIVATE visioncraft_lib project_dependencies GTest::GTest GTest::Main)
target_link_libraries(model_test PRIVATE visioncraft_lib project_dependencies GTest::GTest GTest::Main)


# Register tests with CTest
add_test(NAME MetaVoxelTest COMMAND meta_voxel_test)
add_test(NAME MetaVoxelMapTest COMMAND meta_voxel_map_test)
add_test(NAME ModelTest COMMAND meta_voxel_map_test)
