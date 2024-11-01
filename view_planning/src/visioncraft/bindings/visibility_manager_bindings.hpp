#ifndef VISIBILITY_MANAGER_BINDINGS_HPP
#define VISIBILITY_MANAGER_BINDINGS_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "visioncraft/visibility_manager.h"
#include "bindings_utils.h"

namespace py = pybind11;

inline void bind_visibility_manager(py::module& m) {
    py::class_<visioncraft::VisibilityManager, std::shared_ptr<visioncraft::VisibilityManager>>(m, "VisibilityManager")
        .def(py::init<visioncraft::Model&>(), py::arg("model"))

        // Tracking individual and multiple viewpoints
        .def("trackViewpoint", &visioncraft::VisibilityManager::trackViewpoint, py::arg("viewpoint"))
        .def("untrackViewpoint", &visioncraft::VisibilityManager::untrackViewpoint, py::arg("viewpoint"))
        .def("trackViewpoints", &visioncraft::VisibilityManager::trackViewpoints, py::arg("viewpoints"))
        .def("untrackAllViewpoints", &visioncraft::VisibilityManager::untrackAllViewpoints)

        // Visibility information update
        .def("updateVisibility", &visioncraft::VisibilityManager::updateVisibility, py::arg("viewpoint"))

        // Get visible voxels as a Python set of tuples
        .def("getVisibleVoxels", [](const visioncraft::VisibilityManager& self) {
            py::set visible_voxels;
            for (const auto& key : self.getVisibleVoxels()) {
                visible_voxels.add(py::make_tuple(key.k[0], key.k[1], key.k[2]));
            }
            return visible_voxels;
        })

        // Get visibility count as a Python dictionary
        .def("getVisibilityCount", [](const visioncraft::VisibilityManager& self) {
            py::dict visibility_count;
            for (const auto& pair : self.getVisibilityCount()) {
                py::tuple key = py::make_tuple(pair.first.k[0], pair.first.k[1], pair.first.k[2]);
                visibility_count[key] = pair.second;
            }
            return visibility_count;
        })

        // Get visibility map of viewpoints to observed voxels
        .def("getVisibilityMap", [](const visioncraft::VisibilityManager& self) {
            py::dict visibility_map;
            for (const auto& pair : self.getVisibilityMap()) {
                py::set voxel_set;
                for (const auto& key : pair.second) {
                    voxel_set.add(py::make_tuple(key.k[0], key.k[1], key.k[2]));
                }
                visibility_map[py::cast(pair.first)] = voxel_set;
            }
            return visibility_map;
        })

        // Coverage score retrieval and computation
        .def("getCoverageScore", &visioncraft::VisibilityManager::getCoverageScore)
        .def("computeCoverageScore", &visioncraft::VisibilityManager::computeCoverageScore);
}

#endif // VISIBILITY_MANAGER_BINDINGS_HPP
