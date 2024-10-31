#ifndef META_VOXEL_BINDINGS_HPP
#define META_VOXEL_BINDINGS_HPP

#include <pybind11/pybind11.h>

#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "visioncraft/meta_voxel.h"

#include "bindings_utils.h"


namespace py = pybind11;

inline void bind_meta_voxel(py::module& m) {
    // Expose MetaVoxel class to Python
    py::class_<visioncraft::MetaVoxel>(m, "MetaVoxel")
        .def(py::init([](const Eigen::Vector3d& position, const std::tuple<uint16_t, uint16_t, uint16_t>& octomap_key, float occupancy) {
            // Convert tuple to octomap::OcTreeKey
            octomap::OcTreeKey key;
            key[0] = std::get<0>(octomap_key);
            key[1] = std::get<1>(octomap_key);
            key[2] = std::get<2>(octomap_key);

            return visioncraft::MetaVoxel(position, key, occupancy);
        }), py::arg("position"), py::arg("octomap_key"), py::arg("occupancy") = 0.5)

        .def("getPosition", &visioncraft::MetaVoxel::getPosition, "Retrieve voxel's 3D position.")
        
        .def("getOctomapKey", [](const visioncraft::MetaVoxel& voxel) {
            // Convert OctoMap key to Python tuple (x, y, z)
            const octomap::OcTreeKey& key = voxel.getOctomapKey();
            return std::make_tuple(key[0], key[1], key[2]);
        }, "Retrieve voxel's OctoMap key as a tuple.")
        
        .def("setOccupancy", &visioncraft::MetaVoxel::setOccupancy, py::arg("probability"),
            "Set occupancy probability and update log-odds.")
        
        .def("getOccupancy", &visioncraft::MetaVoxel::getOccupancy, "Get occupancy probability.")
        
        .def("setLogOdds", &visioncraft::MetaVoxel::setLogOdds, py::arg("log_odds"),
            "Set log-odds value and update occupancy.")
        
        .def("getLogOdds", &visioncraft::MetaVoxel::getLogOdds, "Get log-odds value.")
        
        .def("setProperty", [](visioncraft::MetaVoxel& voxel, const std::string& property_name, py::object value) {
            // Convert Python object to the appropriate PropertyValue type
            if (py::isinstance<py::int_>(value)) {
                voxel.setProperty(property_name, value.cast<int>());
            } else if (py::isinstance<py::float_>(value)) {
                voxel.setProperty(property_name, value.cast<float>());
            } else if (py::isinstance<py::str>(value)) {
                voxel.setProperty(property_name, value.cast<std::string>());
            } else if (py::isinstance<py::array>(value)) {
                // Handle Eigen::Vector3d explicitly
                Eigen::Vector3d vec = value.cast<Eigen::Vector3d>();
                voxel.setProperty(property_name, vec);
            } else {
                throw std::runtime_error("Unsupported property type for setProperty.");
            }
        }, py::arg("property_name"), py::arg("value"),
            "Add or update a custom property in the voxel's property map.")
        
        .def("getProperty", [](const visioncraft::MetaVoxel& voxel, const std::string& property_name) -> py::object {
            auto prop = voxel.getProperty(property_name);
            // Convert the variant type to a Python object
            if (auto int_ptr = boost::get<int>(&prop)) {
                return py::int_(*int_ptr);
            } else if (auto float_ptr = boost::get<float>(&prop)) {
                return py::float_(*float_ptr);
            } else if (auto str_ptr = boost::get<std::string>(&prop)) {
                return py::str(*str_ptr);
            } else if (auto vec_ptr = boost::get<Eigen::Vector3d>(&prop)) {
                return py::cast(*vec_ptr);
            } else {
                throw std::runtime_error("Unsupported property type in getProperty.");
            }
        }, py::arg("property_name"),
            "Retrieve a specified property from the voxel's property map.")
        
        .def("hasProperty", &visioncraft::MetaVoxel::hasProperty, py::arg("property_name"),
            "Check if a specified property exists in the voxel's property map.");
}

#endif // META_VOXEL_BINDINGS_HPP
