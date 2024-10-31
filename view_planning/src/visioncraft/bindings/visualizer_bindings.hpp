#ifndef VISUALIZER_BINDINGS_HPP
#define VISUALIZER_BINDINGS_HPP

#include <pybind11/pybind11.h>

#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "bindings_utils.h"

#include "visioncraft/visualizer.h"
namespace py = pybind11;

inline void bind_visualizer(py::module& m) {
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
        .def("addOctomap", &visioncraft::Visualizer::addOctomap, py::arg("model"), py::arg("color") = Eigen::Vector3d(-1, -1, -1))
        .def("addVoxelMap", &visioncraft::Visualizer::addVoxelMap,
            py::arg("model"), py::arg("defaultColor") = Eigen::Vector3d(0.0, 0.0, 1.0))
        .def("addVoxelMapProperty", &visioncraft::Visualizer::addVoxelMapProperty,
            py::arg("model"), py::arg("property_name"),
            py::arg("baseColor") = Eigen::Vector3d(0.0, 1.0, 0.0),
            py::arg("propertyColor") = Eigen::Vector3d(1.0, 1.0, 1.0),
            py::arg("minScale") = -1.0, py::arg("maxScale") = -1.0)
        .def("showGPUVoxelGrid", &visioncraft::Visualizer::showGPUVoxelGrid, py::arg("model"), py::arg("color") = Eigen::Vector3d(1, 0, 0))
        .def("setBackgroundColor", &visioncraft::Visualizer::setBackgroundColor)
        .def("setViewpointFrustumColor", &visioncraft::Visualizer::setViewpointFrustumColor)
        .def("render", &visioncraft::Visualizer::render);

}
#endif // VISUALIZER_BINDINGS_HPP
