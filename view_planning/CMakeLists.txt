cmake_minimum_required(VERSION 3.20)
project(view_planning LANGUAGES CXX CUDA)

# Disable optimization for Debug build and enable debug symbols for C++
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# Disable optimization for Debug build and enable debug symbols for CUDA
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g -G")


# Set paths for Python 3.6
set(PYTHON_EXECUTABLE "/home/abdulaziz/playground/open3d/env/bin/python")
set(PYTHON_LIBRARY "/usr/lib/x86_64-linux-gnu/libpython3.6m.so")
set(PYTHON_INCLUDE_DIR "/usr/include/python3.6m")

# VTK directory
set(VTK_DIR "~/programs/vtk9.1/install/lib/cmake/vtk-9.1")

# CUDA flags, ensure -fPIC is set
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # Globally set -fPIC for all targets

# Find packages
find_package(CGAL REQUIRED)
find_package(octomap REQUIRED)
find_package(Open3D REQUIRED)
find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)
find_package(VTK 9.1 COMPONENTS 
    CommonColor
    CommonCore
    FiltersSources
    InteractionStyle
    RenderingContextOpenGL2
    RenderingCore
    RenderingFreeType
    RenderingGL2PSOpenGL2
    RenderingOpenGL2
    RenderingAnnotation
    Python
    REQUIRED)
find_package(pybind11 REQUIRED)

# Define custom target `Open3D_no_OpenGL` for Open3D without OpenGL-related components
add_library(Open3D_no_OpenGL STATIC IMPORTED)
set_target_properties(Open3D_no_OpenGL PROPERTIES
  IMPORTED_LOCATION "/usr/local/lib/libOpen3D.a"  # Adjust the path to where Open3D is installed
  INTERFACE_COMPILE_DEFINITIONS "_GLIBCXX_USE_CXX11_ABI=1;OPEN3D_STATIC;FMT_HEADER_ONLY=0;FMT_USE_WINDOWS_H=0;FMT_STRING_ALIAS=1"
  INTERFACE_COMPILE_FEATURES "cxx_std_14"
  INTERFACE_LINK_LIBRARIES 
    "\$<LINK_ONLY:stdc++fs>;\$<LINK_ONLY:Open3D::3rdparty_assimp>;\$<LINK_ONLY:Open3D::3rdparty_nanoflann>;\$<LINK_ONLY:Open3D::3rdparty_jpeg>;\$<LINK_ONLY:Open3D::3rdparty_jsoncpp>;\$<LINK_ONLY:Open3D::3rdparty_liblzf>;\$<LINK_ONLY:Open3D::3rdparty_tritriintersect>;\$<LINK_ONLY:Open3D::3rdparty_curl>;\$<LINK_ONLY:Open3D::3rdparty_png>;\$<LINK_ONLY:Open3D::3rdparty_rply>;\$<LINK_ONLY:Open3D::3rdparty_tinyfiledialogs>;\$<LINK_ONLY:Open3D::3rdparty_tinygltf>;\$<LINK_ONLY:Open3D::3rdparty_tinyobjloader>;\$<LINK_ONLY:Open3D::3rdparty_qhullcpp>;\$<LINK_ONLY:Open3D::3rdparty_poisson>;\$<LINK_ONLY:Open3D::3rdparty_imgui>;\$<LINK_ONLY:Open3D::3rdparty_zeromq>;\$<LINK_ONLY:Open3D::3rdparty_msgpack>;\$<LINK_ONLY:Open3D::3rdparty_vtk>;\$<LINK_ONLY:Open3D::3rdparty_uvatlas>;\$<LINK_ONLY:Open3D::3rdparty_tbb>;\$<LINK_ONLY:Open3D::3rdparty_blas>;\$<LINK_ONLY:Open3D::3rdparty_embree>;\$<LINK_ONLY:Open3D::3rdparty_webrtc>;\$<LINK_ONLY:Open3D::3rdparty_civetweb>;\$<LINK_ONLY:Open3D::3rdparty_openmp>;\$<LINK_ONLY:Open3D::3rdparty_ippicv>;Open3D::3rdparty_eigen3;Open3D::3rdparty_parallelstl"
)

# Include directories for dependencies
include_directories(
  ${PYTHON_INCLUDE_DIRS}
  ${VTK_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  "/home/abdulaziz/playground/open3d/env/lib/python3.6/site-packages/pybind11/include"
  include
  include/visioncraft
  "src/visioncraft/bindings" 
)

# Custom target for dependencies
add_library(project_dependencies INTERFACE)
target_include_directories(project_dependencies INTERFACE
  ${PYTHON_INCLUDE_DIRS}
  ${VTK_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(project_dependencies INTERFACE ${VTK_LIBRARIES} Open3D_no_OpenGL ${OCTOMAP_LIBRARIES} Threads::Threads)

# CUDA library for raycasting
add_library(cuda_lib STATIC src/visioncraft/raycasting.cu)
target_link_libraries(cuda_lib PRIVATE ${CUDA_LIBRARIES} Open3D_no_OpenGL)

# Link `cuda_lib` to `project_dependencies`
target_link_libraries(project_dependencies INTERFACE cuda_lib)

# Define source files as a reusable library with -fPIC
add_library(visioncraft_lib STATIC
  src/visioncraft/visualizer.cpp 
  src/visioncraft/model.cpp 
  src/visioncraft/viewpoint.cpp 
  src/visioncraft/meta_voxel_map.cpp
  src/visioncraft/meta_voxel.cpp
  src/visioncraft/visibility_manager.cpp
)

# Link project dependencies to visioncraft_lib
target_link_libraries(visioncraft_lib PRIVATE project_dependencies)

# Executable for visualization test, linking with visioncraft_lib
# add_executable(visualizer_test tests/visualizer_test.cpp)
# target_link_libraries(visualizer_test PRIVATE visioncraft_lib project_dependencies)
add_executable(observer_tests tests/observer_tests.cpp)
target_link_libraries(observer_tests PRIVATE visioncraft_lib project_dependencies)


# Python bindings for visioncraft_py, linking with visioncraft_lib and project_dependencies
pybind11_add_module(visioncraft_py src/visioncraft/bindings/bindings.cpp)

target_link_libraries(visioncraft_py PRIVATE visioncraft_lib project_dependencies ${PYTHON_LIBRARIES})
# Ensure output is placed in the correct folder for Python import with .so extension
set_target_properties(visioncraft_py PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python_bindings
    SUFFIX ".so"
)

# Optional: Enable unit tests
option(ENABLE_UNIT_TESTS "Enable unit tests" OFF)
if (ENABLE_UNIT_TESTS)
    add_subdirectory(tests)
endif()

# Output messages for verification
message(STATUS "Using Python executable: ${PYTHON_EXECUTABLE}")
message(STATUS "Using Python libraries: ${PYTHON_LIBRARIES}")
message(STATUS "Using Python include directories: ${PYTHON_INCLUDE_DIRS}")
