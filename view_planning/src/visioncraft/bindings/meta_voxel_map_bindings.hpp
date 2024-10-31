#ifndef META_VOXEL_MAP_BINDINGS_HPP
#define META_VOXEL_MAP_BINDINGS_HPP

#include <pybind11/pybind11.h>

#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "bindings_utils.h"

#include "visioncraft/meta_voxel_map.h"
namespace py = pybind11;

inline void bind_meta_voxel_map(py::module& m) {
    // Expose MetaVoxelMap class to Python
    py::class_<visioncraft::MetaVoxelMap>(m, "MetaVoxelMap")
        .def(py::init<>())  // Default constructor
        .def("setMetaVoxel", [](visioncraft::MetaVoxelMap &self, py::tuple key, const visioncraft::MetaVoxel& meta_voxel) {
            if (key.size() != 3) throw std::runtime_error("Key tuple must have exactly 3 elements.");
            octomap::OcTreeKey octomap_key(
                key[0].cast<uint16_t>(),
                key[1].cast<uint16_t>(),
                key[2].cast<uint16_t>()
            );
            return self.setMetaVoxel(octomap_key, meta_voxel);
        }, "Insert or update a MetaVoxel in the map using an OctoMap key.")

        // Add the getMap function to provide direct access to the internal map
        .def("getMap", [](visioncraft::MetaVoxelMap &self) -> py::dict {
            py::dict map_dict;
            for (auto& pair : self.getMap()) {
                py::tuple key = py::make_tuple(pair.first.k[0], pair.first.k[1], pair.first.k[2]);
                map_dict[key] = pair.second;  // Assume MetaVoxel is already exposed as a Python class
            }
            return map_dict;
        }, "Retrieve the internal meta voxel map as a Python dictionary.")
        
        .def("getMetaVoxel", [](visioncraft::MetaVoxelMap &self, py::tuple key) -> visioncraft::MetaVoxel* {
            if (key.size() != 3) throw std::runtime_error("Key tuple must have exactly 3 elements.");
            octomap::OcTreeKey octomap_key(
                key[0].cast<uint16_t>(),
                key[1].cast<uint16_t>(),
                key[2].cast<uint16_t>()
            );
            return self.getMetaVoxel(octomap_key);
        }, py::return_value_policy::reference, "Retrieve a MetaVoxel instance using an OctoMap key.")
        .def("setPropertyForAllVoxels", [](visioncraft::MetaVoxelMap &self, const std::string& property_name, py::object value) {
            visioncraft::MetaVoxel::PropertyValue prop_value;

            // Determine the type of the Python object and cast it to the appropriate variant type
            if (py::isinstance<py::int_>(value)) {
                prop_value = value.cast<int>();
            } else if (py::isinstance<py::float_>(value)) {
                prop_value = value.cast<float>();
            } else if (py::isinstance<py::str>(value)) {
                prop_value = value.cast<std::string>();
            } else if (py::isinstance<py::array>(value)) {
                // Handle Eigen::Vector3d explicitly if value is a numpy array
                Eigen::Vector3d vec = value.cast<Eigen::Vector3d>();
                prop_value = vec;
            } else {
                throw std::runtime_error("Unsupported property type for setPropertyForAllVoxels.");
            }

            // Call the C++ function with the converted property value
            return self.setPropertyForAllVoxels(property_name, prop_value);
        }, py::arg("property_name"), py::arg("value"), "Set a specified property with a given value for all MetaVoxel instances in the map.")
        .def("setMetaVoxelProperty", [](visioncraft::MetaVoxelMap &self, py::tuple key, const std::string& property_name, py::object value) {
            if (key.size() != 3) throw std::runtime_error("Key tuple must have exactly 3 elements.");
            octomap::OcTreeKey octomap_key(
                key[0].cast<uint16_t>(),
                key[1].cast<uint16_t>(),
                key[2].cast<uint16_t>()
            );

            if (py::isinstance<py::int_>(value)) {
                return self.setMetaVoxelProperty(octomap_key, property_name, value.cast<int>());
            } else if (py::isinstance<py::float_>(value)) {
                return self.setMetaVoxelProperty(octomap_key, property_name, value.cast<float>());
            } else if (py::isinstance<py::str>(value)) {
                return self.setMetaVoxelProperty(octomap_key, property_name, value.cast<std::string>());
            } else if (py::isinstance<py::array>(value)) {
                Eigen::Vector3d vec = value.cast<Eigen::Vector3d>();
                return self.setMetaVoxelProperty(octomap_key, property_name, vec);
            } else {
                throw std::runtime_error("Unsupported property type for setMetaVoxelProperty.");
            }
        }, "Set a custom property for a MetaVoxel using an OctoMap key.")

        .def("getMetaVoxelProperty", [](const visioncraft::MetaVoxelMap &self, py::tuple key, const std::string& property_name) -> py::object {
            if (key.size() != 3) throw std::runtime_error("Key tuple must have exactly 3 elements.");
            octomap::OcTreeKey octomap_key(
                key[0].cast<uint16_t>(),
                key[1].cast<uint16_t>(),
                key[2].cast<uint16_t>()
            );

            auto prop = self.getMetaVoxelProperty(octomap_key, property_name);
            if (auto int_ptr = boost::get<int>(&prop)) {
                return py::int_(*int_ptr);
            } else if (auto float_ptr = boost::get<float>(&prop)) {
                return py::float_(*float_ptr);
            } else if (auto str_ptr = boost::get<std::string>(&prop)) {
                return py::str(*str_ptr);
            } else if (auto vec_ptr = boost::get<Eigen::Vector3d>(&prop)) {
                return py::cast(*vec_ptr);
            } else {
                throw std::runtime_error("Unsupported property type in getMetaVoxelProperty.");
            }
        }, "Retrieve a custom property from a MetaVoxel using an OctoMap key.")

        .def("clear", &visioncraft::MetaVoxelMap::clear, "Clear all MetaVoxel entries in the map.")
        .def("size", &visioncraft::MetaVoxelMap::size, "Get the size of the MetaVoxel map.");
}
#endif // META_VOXEL_MAP_BINDINGS_HPP
