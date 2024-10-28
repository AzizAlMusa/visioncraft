cmake_minimum_required(VERSION 3.20)
project(view_planning LANGUAGES CXX CUDA)

# Set default build type
# if(NOT CMAKE_BUILD_TYPE)
#   set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build: Debug Release RelWithDebInfo MinSizeRel." FORCE)
# endif()

# Set Python executable and libraries explicitly for Python 3.6
set(PYTHON_EXECUTABLE "/home/abdulaziz/playground/open3d/env/bin/python")
set(PYTHON_LIBRARY "/usr/lib/x86_64-linux-gnu/libpython3.6m.so")
set(PYTHON_INCLUDE_DIR "/usr/include/python3.6m")


set(VTK_DIR "~/programs/vtk9.1/install/lib/cmake/vtk-9.1")



# Find necessary libraries
find_package(CGAL REQUIRED)
find_package(octomap REQUIRED)
find_package(Open3D REQUIRED)
find_package(CUDA REQUIRED)


# Add the -fPIC flag for CUDA
# This is done so that the python bindings compile, otherwise it fails
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")


# Define a custom target `Open3D_no_OpenGL` to use Open3D without OpenGL-related components
# VTK cannot work with Open3D when they are linked together in the same target, each will have its own OpenGL context and it causes segfaults
add_library(Open3D_no_OpenGL STATIC IMPORTED)
# Set target properties based on your specific needs
# this was taken from Open3DTargets.cmake file and removing:
# \$<LINK_ONLY:Open3D::3rdparty_glew>, \$<LINK_ONLY:Open3D::3rdparty_glfw>, and \$<LINK_ONLY:Open3D::3rdparty_opengl>.
set_target_properties(Open3D_no_OpenGL PROPERTIES
  IMPORTED_LOCATION "/usr/local/lib/libOpen3D.a"  # Adjust the path to where Open3D is installed
  INTERFACE_COMPILE_DEFINITIONS "_GLIBCXX_USE_CXX11_ABI=1;OPEN3D_STATIC;FMT_HEADER_ONLY=0;FMT_USE_WINDOWS_H=0;FMT_STRING_ALIAS=1"
  INTERFACE_COMPILE_FEATURES "cxx_std_14"
  INTERFACE_INCLUDE_DIRECTORIES "${Open3D_INCLUDE_DIRS}"  # Use the found include directories
  INTERFACE_LINK_LIBRARIES 
    "\$<LINK_ONLY:stdc++fs>;\$<LINK_ONLY:Open3D::3rdparty_assimp>;\$<LINK_ONLY:Open3D::3rdparty_nanoflann>;\$<LINK_ONLY:Open3D::3rdparty_jpeg>;\$<LINK_ONLY:Open3D::3rdparty_jsoncpp>;\$<LINK_ONLY:Open3D::3rdparty_liblzf>;\$<LINK_ONLY:Open3D::3rdparty_tritriintersect>;\$<LINK_ONLY:Open3D::3rdparty_curl>;\$<LINK_ONLY:Open3D::3rdparty_png>;\$<LINK_ONLY:Open3D::3rdparty_rply>;\$<LINK_ONLY:Open3D::3rdparty_tinyfiledialogs>;\$<LINK_ONLY:Open3D::3rdparty_tinygltf>;\$<LINK_ONLY:Open3D::3rdparty_tinyobjloader>;\$<LINK_ONLY:Open3D::3rdparty_qhullcpp>;\$<LINK_ONLY:Open3D::3rdparty_poisson>;\$<LINK_ONLY:Open3D::3rdparty_imgui>;\$<LINK_ONLY:Open3D::3rdparty_zeromq>;\$<LINK_ONLY:Open3D::3rdparty_msgpack>;\$<LINK_ONLY:Open3D::3rdparty_vtk>;\$<LINK_ONLY:Open3D::3rdparty_uvatlas>;\$<LINK_ONLY:Open3D::3rdparty_tbb>;\$<LINK_ONLY:Open3D::3rdparty_blas>;\$<LINK_ONLY:Open3D::3rdparty_embree>;\$<LINK_ONLY:Open3D::3rdparty_webrtc>;\$<LINK_ONLY:Open3D::3rdparty_civetweb>;\$<LINK_ONLY:Open3D::3rdparty_openmp>;\$<LINK_ONLY:Open3D::3rdparty_ippicv>;Open3D::3rdparty_eigen3;Open3D::3rdparty_parallelstl"
)




find_package(Threads REQUIRED)
# find_package(VTK REQUIRED)
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

# Find Python 3.6 libraries and headers
find_package(PythonLibs 3.6 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
link_directories(${PYTHON_LIBRARIES})

# Re-enable pybind11 package and point to Python 3.6 version
find_package(pybind11 REQUIRED)

# Manually add pybind11 include path for Python 3.6
include_directories("/home/abdulaziz/playground/open3d/env/lib/python3.6/site-packages/pybind11/include")

# Include directories for other dependencies
include_directories(
  ${VTK_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  include
  include/visioncraft
)


# Executable for testing model_loader (C++ only)
# add_executable(exploration_simulator src/visioncraft/exploration_simulator.cpp src/visioncraft/model_loader.cpp src/visioncraft/viewpoint.cpp)
# target_link_libraries(exploration_simulator ${OCTOMAP_LIBRARIES} Open3D::Open3D Threads::Threads)

# # Executable for testing model_loader (C++ only)
# add_executable(model_loader_tests tests/model_loader_tests.cpp src/visioncraft/model_loader.cpp)
# target_link_libraries(model_loader_tests ${OCTOMAP_LIBRARIES} Open3D::Open3D Threads::Threads)


# add_executable(first_simulator tests/exploration_view_tests.cpp src/visioncraft/model_loader.cpp src/visioncraft/viewpoint.cpp)
# target_link_libraries(first_simulator ${OCTOMAP_LIBRARIES} Open3D::Open3D Threads::Threads)


# add_executable(nbv_input_visualizer tests/nbv_input_visualizer.cpp)
# target_link_libraries(nbv_input_visualizer ${VTK_LIBRARIES} ${PCL_LIBRARIES})

# Executable for testing model_loader (C++ only)


add_library(cuda_lib src/visioncraft/raycasting.cu) 
target_link_libraries(cuda_lib PRIVATE ${CUDA_LIBRARIES} Open3D_no_OpenGL)

add_executable(visualizer_test tests/visualizer_test.cpp  
src/visioncraft/visualizer.cpp 
src/visioncraft/model_loader.cpp 
src/visioncraft/viewpoint.cpp 
src/visioncraft/meta_voxel_map.cpp
src/visioncraft/meta_voxel.cpp)
target_link_libraries(visualizer_test PRIVATE ${VTK_LIBRARIES} Open3D_no_OpenGL ${OCTOMAP_LIBRARIES} Threads::Threads  ${CUDA_LIBRARIES} cuda_lib)
# add_executable(visualizer_test tests/visualizer_test.cpp  )
# target_link_libraries(visualizer_test PRIVATE ${VTK_LIBRARIES} )


# Python bindings 
pybind11_add_module(visioncraft_py src/visioncraft/bindings.cpp src/visioncraft/model_loader.cpp src/visioncraft/viewpoint.cpp src/visioncraft/visualizer.cpp src/visioncraft/meta_voxel_map.cpp src/visioncraft/meta_voxel.cpp)

# Link necessary libraries, including Python 3.6 libraries
target_link_libraries(visioncraft_py PRIVATE ${VTK_LIBRARIES} ${OCTOMAP_LIBRARIES} Open3D_no_OpenGL Threads::Threads ${PYTHON_LIBRARIES} cuda_lib)

# Ensure output is placed in the correct folder for Python import with .so extension
set_target_properties(visioncraft_py PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python_bindings
    SUFFIX ".so"
)

# Optionally, output useful messages to verify the correct Python interpreter and libraries
message(STATUS "Using Python executable: ${PYTHON_EXECUTABLE}")
message(STATUS "Using Python libraries: ${PYTHON_LIBRARIES}")
message(STATUS "Using Python include directories: ${PYTHON_INCLUDE_DIRS}")


message(STATUS "Open3D_INCLUDE_DIRS: ${Open3D_INCLUDE_DIRS}")
message(STATUS "Open3D_LIBRARIES: ${Open3D_LIBRARIES}")
