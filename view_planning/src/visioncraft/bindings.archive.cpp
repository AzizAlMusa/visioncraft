#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "visioncraft/model_loader.h"
#include "visioncraft/viewpoint.h"

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
        .def("generateExplorationMap", py::overload_cast<int, const octomap::point3d&, const octomap::point3d&>(&visioncraft::ModelLoader::generateExplorationMap));

    // Expose the Viewpoint class to Python
    py::class_<visioncraft::Viewpoint>(m, "Viewpoint")
        .def(py::init<>())  // Default constructor
        .def(py::init<const Eigen::Vector3d&, const Eigen::Matrix3d&, double, double, int, int, double, double>())
        .def(py::init<const Eigen::VectorXd&, double, double, int, int, double, double>())
        .def(py::init<const Eigen::Vector3d&, const Eigen::Vector3d&, const Eigen::Vector3d&, double, double, int, int, double, double>())
        .def("setPosition", &visioncraft::Viewpoint::setPosition)
        .def("getPosition", &visioncraft::Viewpoint::getPosition)
        .def("setOrientation", py::overload_cast<const Eigen::Matrix3d&>(&visioncraft::Viewpoint::setOrientation))
        .def("setOrientation", py::overload_cast<const Eigen::Quaterniond&>(&visioncraft::Viewpoint::setOrientation))
        .def("setLookAt", &visioncraft::Viewpoint::setLookAt)
        .def("getOrientationMatrix", &visioncraft::Viewpoint::getOrientationMatrix)
        .def("getOrientationQuaternion", &visioncraft::Viewpoint::getOrientationQuaternion)
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
        .def("performRaycasting", &visioncraft::Viewpoint::performRaycasting);
}
