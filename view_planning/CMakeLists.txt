# Set the minimum required CMake version and define the project
cmake_minimum_required(VERSION 3.20)
project(view_planning)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

# Find external libraries like PCL, VTK, CGAL, OctoMap, mlpack, and CUDA
find_package(CGAL REQUIRED)
include(${CGAL_USE_FILE})

find_package(octomap REQUIRED)
find_package(Open3D REQUIRED)
find_package(Threads REQUIRED)  # Threading support in docker


find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

# Include directories for header files
include_directories(
  ${VTK_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
  include
  include/visioncraft  # Add this line to include your library headers

)

# Define an executable and link libraries for the executable
add_executable(model_loader_tests tests/model_loader_tests.cpp src/visioncraft/model_loader.cpp)
target_link_libraries(model_loader_tests  ${OCTOMAP_LIBRARIES} Open3D::Open3D Threads::Threads)
# add_dependencies(model_loader_tests ${${PROJECT_NAME}_EXPORTED_TARGETS} )

add_executable(viewpoint_tests tests/viewpoint_tests.cpp src/visioncraft/viewpoint.cpp)
target_link_libraries(viewpoint_tests   ${OCTOMAP_LIBRARIES} Open3D::Open3D Threads::Threads)
# add_dependencies(viewpoint_tests ${${PROJECT_NAME}_EXPORTED_TARGETS})

# # Define an executable and link libraries for the executable
# add_executable(visualizer_tests tests/visualizer_tests.cpp src/visioncraft/visualizer.cpp src/visioncraft/viewpoint.cpp src/visioncraft/model_loader.cpp )
# target_link_libraries(visualizer_tests ${VTK_LIBRARIES} ${OCTOMAP_LIBRARIES} Open3D::Open3D )
# add_dependencies(visualizer_tests ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# add_executable(rviz_visualizer_tests tests/rviz_visualizer_tests.cpp src/visioncraft/rviz_visualizer.cpp src/visioncraft/viewpoint.cpp src/visioncraft/model_loader.cpp)
# target_link_libraries(rviz_visualizer_tests ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${VTK_LIBRARIES} ${OCTOMAP_LIBRARIES} Open3D::Open3D ${CUDA_LIBRARIES} ${rviz_visual_tools_LIBRARIES})  # Link rviz_visual_tools
# add_dependencies(rviz_visualizer_tests ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# CUDA_ADD_LIBRARY(CudaTesterLib STATIC src/visioncraft/my_cuda_code.cu)
# add_executable(cuda_tester tests/cuda_tester.cpp)
# target_link_libraries(cuda_tester ${CUDA_LIBRARIES} CudaTesterLib)
# add_dependencies(cuda_tester ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(exploration_view_tests tests/exploration_view_tests.cpp src/visioncraft/model_loader.cpp src/visioncraft/viewpoint.cpp) #src/visioncraft/model_loader.cpp src/visioncraft/viewpoint.cpp src/visioncraft/rviz_visualizer.cpp
target_link_libraries(exploration_view_tests   ${OCTOMAP_LIBRARIES} Open3D::Open3D Threads::Threads)
# add_dependencies(exploration_view_tests ${${PROJECT_NAME}_EXPORTED_TARGETS})

add_executable(nbv_input_visualizer tests/nbv_input_visualizer.cpp ) #src/visioncraft/model_loader.cpp src/visioncraft/viewpoint.cpp src/visioncraft/rviz_visualizer.cpp
target_link_libraries(nbv_input_visualizer  ${VTK_LIBRARIES} ${PCL_LIBRARIES} )
# add_dependencies(nbv_input_visualizer ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})