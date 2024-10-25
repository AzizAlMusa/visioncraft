#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "visioncraft/model_loader.h"
#include "visioncraft/viewpoint.h"
#include "visioncraft/visualizer.h"

namespace py = pybind11;

PYBIND11_MODULE(visioncraft_py, m) {

   

    // Expose the ModelLoader class to Python
    py::class_<visioncraft::ModelLoader>(m, "ModelLoader")
        .def(py::init<>())  // Expose the constructor
        .def("loadMesh", &visioncraft::ModelLoader::loadMesh)
        .def("loadModel", &visioncraft::ModelLoader::loadModel)
        .def("loadExplorationModel", &visioncraft::ModelLoader::loadExplorationModel)
        .def("initializeRaycastingScene", &visioncraft::ModelLoader::initializeRaycastingScene)
        .def("generatePointCloud", &visioncraft::ModelLoader::generatePointCloud)
        .def("correctNormalsUsingSignedDistance", &visioncraft::ModelLoader::correctNormalsUsingSignedDistance)
        .def("generateVolumetricPointCloud", &visioncraft::ModelLoader::generateVolumetricPointCloud)
        .def("generateVoxelGrid", &visioncraft::ModelLoader::generateVoxelGrid)
        .def("generateOctoMap", &visioncraft::ModelLoader::generateOctoMap)
        .def("generateVolumetricOctoMap", &visioncraft::ModelLoader::generateVolumetricOctoMap)
        .def("generateSurfaceShellOctomap", &visioncraft::ModelLoader::generateSurfaceShellOctomap)
        .def("generateExplorationMap", py::overload_cast<double, const octomap::point3d&, const octomap::point3d&>(&visioncraft::ModelLoader::generateExplorationMap))
        .def("generateExplorationMap", py::overload_cast<int, const octomap::point3d&, const octomap::point3d&>(&visioncraft::ModelLoader::generateExplorationMap))
        .def("convertVoxelGridToGPUFormat", &visioncraft::ModelLoader::convertVoxelGridToGPUFormat)  
        .def("getGPUVoxelGrid", &visioncraft::ModelLoader::getGPUVoxelGrid)  
        .def("updateVoxelGridFromHits", &visioncraft::ModelLoader::updateVoxelGridFromHits) 
        .def("updateOctomapWithHits", &visioncraft::ModelLoader::updateOctomapWithHits);  

     // Expose VoxelGridGPU to Python
    py::class_<visioncraft::VoxelGridGPU>(m, "VoxelGridGPU")
        .def(py::init<>())  // Default constructor
        .def("get_voxel_data", [](const visioncraft::VoxelGridGPU& self) {
            return std::vector<int>(self.voxel_data, self.voxel_data + (self.width * self.height * self.depth));
        })
        .def_readonly("width", &visioncraft::VoxelGridGPU::width)
        .def_readonly("height", &visioncraft::VoxelGridGPU::height)
        .def_readonly("depth", &visioncraft::VoxelGridGPU::depth)
        .def_readonly("voxel_size", &visioncraft::VoxelGridGPU::voxel_size)
        .def_readonly("min_bound", &visioncraft::VoxelGridGPU::min_bound);

    // Expose the Viewpoint class to Python
    py::class_<visioncraft::Viewpoint>(m, "Viewpoint")
        .def(py::init<>())  // Default constructor
        .def(py::init<const Eigen::Vector3d&, const Eigen::Matrix3d&, double, double, int, int, double, double>())
        .def(py::init<const Eigen::VectorXd&, double, double, int, int, double, double>())
        .def(py::init<const Eigen::Vector3d&, const Eigen::Vector3d&, const Eigen::Vector3d&, double, double, int, int, double, double>())
        .def("setPosition", &visioncraft::Viewpoint::setPosition)
        .def("getPosition", &visioncraft::Viewpoint::getPosition)
        .def("setOrientation", py::overload_cast<const Eigen::Matrix3d&>(&visioncraft::Viewpoint::setOrientation))
        .def("setOrientation", [](visioncraft::Viewpoint& self, py::array_t<double> quat_array) {
            // Ensure the array has the correct shape (1D array of size 4)
            if (quat_array.ndim() != 1 || quat_array.shape(0) != 4) {
                throw std::runtime_error("Quaternion array must have exactly 4 elements in a 1D array.");
            }

            // Access the raw data safely
            double* buf = quat_array.mutable_data();

            // Create the Eigen Quaternion using the data from the NumPy array
            Eigen::Quaterniond quat(buf[0], buf[1], buf[2], buf[3]);

            // Set the orientation in the Viewpoint instance
            self.setOrientation(quat);
        })
        .def("setLookAt", &visioncraft::Viewpoint::setLookAt)
        .def("getOrientationMatrix", &visioncraft::Viewpoint::getOrientationMatrix)
        .def("getOrientationQuaternion", [](const visioncraft::Viewpoint& self) {
            // Retrieve the quaternion from the Viewpoint instance
            Eigen::Quaterniond quat = self.getOrientationQuaternion();

            // Create a NumPy array with 4 elements (w, x, y, z)
            py::array_t<double> result({4});
            
            // Fill the array with quaternion components
            auto r = result.mutable_unchecked<1>();
            r(0) = quat.w();
            r(1) = quat.x();
            r(2) = quat.y();
            r(3) = quat.z();
            
            return result;
        })
        .def("getTransformationMatrix", &visioncraft::Viewpoint::getTransformationMatrix)
        .def("getOrientationEuler", &visioncraft::Viewpoint::getOrientationEuler)
        .def("setNearPlane", &visioncraft::Viewpoint::setNearPlane)
        .def("getNearPlane", &visioncraft::Viewpoint::getNearPlane)
        .def("setFarPlane", &visioncraft::Viewpoint::setFarPlane)
        .def("getFarPlane", &visioncraft::Viewpoint::getFarPlane)
        .def("setResolution", &visioncraft::Viewpoint::setResolution)
        .def("getResolution", &visioncraft::Viewpoint::getResolution)
        .def("setDownsampleFactor", &visioncraft::Viewpoint::setDownsampleFactor)
        .def("getDownsampleFactor", &visioncraft::Viewpoint::getDownsampleFactor)
        .def("getDownsampledResolution", &visioncraft::Viewpoint::getDownsampledResolution)
        .def("setHorizontalFieldOfView", &visioncraft::Viewpoint::setHorizontalFieldOfView)
        .def("getHorizontalFieldOfView", &visioncraft::Viewpoint::getHorizontalFieldOfView)
        .def("setVerticalFieldOfView", &visioncraft::Viewpoint::setVerticalFieldOfView)
        .def("getVerticalFieldOfView", &visioncraft::Viewpoint::getVerticalFieldOfView) 
        .def("getFrustumCorners", &visioncraft::Viewpoint::getFrustumCorners)
        .def("generateRays", &visioncraft::Viewpoint::generateRays)
        .def("performRaycasting", [](visioncraft::Viewpoint &self, const visioncraft::ModelLoader& modelLoader, bool use_parallel) {
            // Call the original performRaycasting method
            auto result = self.performRaycasting(modelLoader, use_parallel);

            // Create a Python dictionary to hold the result
            py::dict py_result;

            // Iterate over the unordered_map and convert keys to Python tuples
            for (const auto& item : result) {
                const octomap::OcTreeKey& key = item.first;
                bool value = item.second;

                // Convert octomap::OcTreeKey to a Python tuple (x, y, z)
                py::tuple py_key = py::make_tuple(key.k[0], key.k[1], key.k[2]);

                // Add the key-value pair to the Python dictionary
                py_result[py_key] = value;
            }

            return py_result;
        })
        .def("performRaycastingOnGPU", &visioncraft::Viewpoint::performRaycastingOnGPU);


        // Expose the Visualizer class to Python
        py::class_<visioncraft::Visualizer>(m, "Visualizer")
            .def(py::init<>())  // Expose the constructor
            .def("initializeWindow", &visioncraft::Visualizer::initializeWindow)
            .def("addViewpoint", &visioncraft::Visualizer::addViewpoint, py::arg("viewpoint"), py::arg("showFrustum") = true, py::arg("showAxes") = true)
            .def("addMultipleViewpoints", &visioncraft::Visualizer::addMultipleViewpoints)
            .def("showRays", &visioncraft::Visualizer::showRays)
            .def("showRayVoxels", &visioncraft::Visualizer::showRayVoxels)
            .def("showViewpointHits", &visioncraft::Visualizer::showViewpointHits)
            .def("addMesh", &visioncraft::Visualizer::addMesh)
            .def("addPointCloud", &visioncraft::Visualizer::addPointCloud)
            .def("addOctomap", &visioncraft::Visualizer::addOctomap, py::arg("modelLoader"), py::arg("color") = Eigen::Vector3d(-1, -1, -1))
            .def("showGPUVoxelGrid", &visioncraft::Visualizer::showGPUVoxelGrid, py::arg("modelLoader"), py::arg("color") = Eigen::Vector3d(1, 0, 0))
            .def("setBackgroundColor", &visioncraft::Visualizer::setBackgroundColor)
            .def("setViewpointFrustumColor", &visioncraft::Visualizer::setViewpointFrustumColor)
            .def("render", &visioncraft::Visualizer::render);

        // py::class_<visioncraft::ExplorationSimulator>(m, "ExplorationSimulator")
        // .def(py::init<>())
        // .def("loadModel", &visioncraft::ExplorationSimulator::loadModel, py::arg("model_file"), py::arg("num_points") = 20000, py::arg("grid_resolution") = 32)
        // .def("setViewpoints", &visioncraft::ExplorationSimulator::setViewpoints)
        // .def("performRaycasting", &visioncraft::ExplorationSimulator::performRaycasting)
        // .def("getExplorationMapData", &visioncraft::ExplorationSimulator::getExplorationMapData)
        // .def("getCoverageScore", &visioncraft::ExplorationSimulator::getCoverageScore)
        // .def("getScalingFactor", &visioncraft::ExplorationSimulator::getScalingFactor);
}

