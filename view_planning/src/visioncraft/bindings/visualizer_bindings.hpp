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
    py::class_<visioncraft::Visualizer>(m, "Visualizer")
        .def(py::init<>())  // Constructor

        // Initialization and Rendering
        .def("initializeWindow", &visioncraft::Visualizer::initializeWindow, py::arg("windowName") = "3D Visualizer")
        .def("processEvents", &visioncraft::Visualizer::processEvents)
        .def("render", &visioncraft::Visualizer::render)
        .def("renderStep", &visioncraft::Visualizer::renderStep)
        .def("startAsyncRendering", &visioncraft::Visualizer::startAsyncRendering)
        .def("stopAsyncRendering", &visioncraft::Visualizer::stopAsyncRendering)
        .def("getRenderWindowInteractor", &visioncraft::Visualizer::getRenderWindowInteractor)
        .def("getRenderer", &visioncraft::Visualizer::getRenderer)

        // Viewpoint Management
        .def("addViewpoint", &visioncraft::Visualizer::addViewpoint, py::arg("viewpoint"), py::arg("showFrustum") = true, py::arg("showAxes") = true, py::arg("showPosition") = false, py::arg("showDirection") = false)
        .def("updateViewpoint", &visioncraft::Visualizer::updateViewpoint, py::arg("viewpoint"), py::arg("updateFrustum"), py::arg("updateAxes"), py::arg("updatePosition"), py::arg("updateDirection"))
        .def("addMultipleViewpoints", &visioncraft::Visualizer::addMultipleViewpoints)
        .def("removeViewpoints", &visioncraft::Visualizer::removeViewpoints)
        .def("removeViewpoint", &visioncraft::Visualizer::removeViewpoint)

        // Ray Visualization
        .def("showRays", &visioncraft::Visualizer::showRays)
        .def("showRaysParallel", &visioncraft::Visualizer::showRaysParallel)
        .def("showRayVoxels", &visioncraft::Visualizer::showRayVoxels)

        // Viewpoint Hit Visualization
        .def("showViewpointHits", &visioncraft::Visualizer::showViewpointHits)

        // 3D Object Visualization
        .def("addMesh", &visioncraft::Visualizer::addMesh)
        .def("addPointCloud", &visioncraft::Visualizer::addPointCloud)
        .def("addOctomap", &visioncraft::Visualizer::addOctomap, py::arg("model"), py::arg("color") = Eigen::Vector3d(-1, -1, -1))
        .def("showGPUVoxelGrid", &visioncraft::Visualizer::showGPUVoxelGrid, py::arg("model"), py::arg("color") = Eigen::Vector3d(1, 0, 0))
        .def("addVoxelMap", &visioncraft::Visualizer::addVoxelMap, py::arg("model"), py::arg("defaultColor") = Eigen::Vector3d(0.0, 0.0, 1.0))
        .def("addVoxelMapProperty", &visioncraft::Visualizer::addVoxelMapProperty, py::arg("model"), py::arg("property_name"), py::arg("baseColor") = Eigen::Vector3d(0.0, 1.0, 0.0), py::arg("propertyColor") = Eigen::Vector3d(1.0, 1.0, 1.0), py::arg("minScale") = -1.0, py::arg("maxScale") = -1.0)
        .def("removeVoxelMapProperty", &visioncraft::Visualizer::removeVoxelMapProperty)

        // Background and Frustum Colors
        .def("setBackgroundColor", &visioncraft::Visualizer::setBackgroundColor)
        .def("setViewpointFrustumColor", &visioncraft::Visualizer::setViewpointFrustumColor)

        // Overlay Text
        .def("addOverlayText", &visioncraft::Visualizer::addOverlayText, py::arg("text"), py::arg("x"), py::arg("y"), py::arg("fontSize"), py::arg("color"))
        .def("removeOverlayTexts", &visioncraft::Visualizer::removeOverlayTexts);
}

#endif // VISUALIZER_BINDINGS_HPP
