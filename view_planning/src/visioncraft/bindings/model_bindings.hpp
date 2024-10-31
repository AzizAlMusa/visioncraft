#ifndef MODEL_BINDINGS_HPP
#define MODEL_BINDINGS_HPP

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "bindings_utils.h"

#include "visioncraft/model.h"


namespace py = pybind11;

inline void bind_model(py::module& m) {
    py::class_<visioncraft::Model>(m, "Model")
        .def(py::init<>())
        // Load functions
        .def("loadMesh", &visioncraft::Model::loadMesh)
        .def("loadModel", &visioncraft::Model::loadModel, py::arg("file_path"), py::arg("num_samples") = 10000, py::arg("resolution") = -1)
        .def("loadExplorationModel", &visioncraft::Model::loadExplorationModel, py::arg("file_path"), py::arg("num_samples") = 10000, py::arg("num_cells_per_side") = 32)
        .def("initializeRaycastingScene", &visioncraft::Model::initializeRaycastingScene)

        // Point cloud functions
        .def("generatePointCloud", &visioncraft::Model::generatePointCloud)
        .def("getPointCloud", &visioncraft::Model::getPointCloud, py::return_value_policy::reference)
        .def("getAverageSpacing", &visioncraft::Model::getAverageSpacing)

        // Bounding box functions
        .def("getMinBound", [](const visioncraft::Model& self) {
            const octomap::point3d& min_bound = self.getMinBound();
            return Eigen::Vector3d(min_bound.x(), min_bound.y(), min_bound.z());
        })
        .def("getMaxBound", [](const visioncraft::Model& self) {
            const octomap::point3d& max_bound = self.getMaxBound();
            return Eigen::Vector3d(max_bound.x(), max_bound.y(), max_bound.z());
        })
        .def("getCenter", [](const visioncraft::Model& self) {
            const octomap::point3d& center = self.getCenter();
            return Eigen::Vector3d(center.x(), center.y(), center.z());
        })

        // Mesh data functions
        .def("getMeshData", &visioncraft::Model::getMeshData, py::return_value_policy::reference)
        .def("correctNormalsUsingSignedDistance", &visioncraft::Model::correctNormalsUsingSignedDistance)

        // Voxel grid functions
        .def("generateVoxelGrid", &visioncraft::Model::generateVoxelGrid)
        .def("getVoxelGrid", &visioncraft::Model::getVoxelGrid, py::return_value_policy::reference)

        // Octomap functions
        .def("generateOctoMap", &visioncraft::Model::generateOctoMap)
        .def("getOctomap", &visioncraft::Model::getOctomap, py::return_value_policy::reference)
        .def("generateVolumetricOctoMap", &visioncraft::Model::generateVolumetricOctoMap)
        .def("getVolumetricOctomap", &visioncraft::Model::getVolumetricOctomap, py::return_value_policy::reference)
        .def("generateSurfaceShellOctomap", &visioncraft::Model::generateSurfaceShellOctomap)
        .def("getSurfaceShellOctomap", &visioncraft::Model::getSurfaceShellOctomap, py::return_value_policy::reference)

        // Exploration map functions
        .def("generateExplorationMap", py::overload_cast<double, const octomap::point3d&, const octomap::point3d&>(&visioncraft::Model::generateExplorationMap))
        .def("generateExplorationMap", py::overload_cast<int, const octomap::point3d&, const octomap::point3d&>(&visioncraft::Model::generateExplorationMap))
        .def("getExplorationMap", &visioncraft::Model::getExplorationMap, py::return_value_policy::reference)

        // GPU voxel grid functions
        .def("convertVoxelGridToGPUFormat", &visioncraft::Model::convertVoxelGridToGPUFormat)
        // Bind convertGPUHitsToOctreeKeys
        .def("convertGPUHitsToOctreeKeys", [](const visioncraft::Model& self, const std::set<std::tuple<int, int, int>>& unique_hit_voxels) {
            auto octree_hits = self.convertGPUHitsToOctreeKeys(unique_hit_voxels);

            // Convert the unordered_map of OcTreeKeys to a Python dictionary
            py::dict hits_dict;
            for (auto it = octree_hits.begin(); it != octree_hits.end(); ++it) {
                hits_dict[keyToTuple(it->first)] = it->second;
            }
            return hits_dict;
        }, py::arg("unique_hit_voxels"))
        .def("getGPUVoxelGrid", &visioncraft::Model::getGPUVoxelGrid)
        .def("updateVoxelGridFromHits", &visioncraft::Model::updateVoxelGridFromHits)
        .def("updateOctomapWithHits", &visioncraft::Model::updateOctomapWithHits)

        // MetaVoxel Map Functions
        .def("generateVoxelMap", &visioncraft::Model::generateVoxelMap)

        // MetaVoxel property functions
        .def("getVoxel", [](visioncraft::Model& self, const py::tuple& key_tuple) {
            return self.getVoxel(tupleToKey(key_tuple));
        }, py::return_value_policy::reference)
        .def("getVoxel", [](visioncraft::Model& self, const Eigen::Vector3d& position) {
            return self.getVoxel(position);
        }, py::return_value_policy::reference)
        .def("getVoxelMap", [](const visioncraft::Model& self) {
            py::dict meta_voxel_dict;
            const auto& meta_voxel_map = self.getVoxelMap().getMap();

            for (const auto& item : meta_voxel_map) {
                const auto& key = item.first;
                const auto& voxel = item.second;
                meta_voxel_dict[keyToTuple(key)] = voxel;
            }
            return meta_voxel_dict;
        })
        .def("updateVoxelOccupancy", 
            [](visioncraft::Model& self, const py::tuple& key_tuple, float new_occupancy) {
                return self.updateVoxelOccupancy(tupleToKey(key_tuple), new_occupancy);
            }, py::arg("key"), py::arg("new_occupancy"))
        .def("updateVoxelOccupancy", 
            [](visioncraft::Model& self, const Eigen::Vector3d& position, float new_occupancy) {
                return self.updateVoxelOccupancy(position, new_occupancy);
            }, py::arg("position"), py::arg("new_occupancy"))
        .def("addVoxelProperty", [](visioncraft::Model& self, const std::string& property_name, py::object value) {
            visioncraft::MetaVoxel::PropertyValue prop_value;

            // Determine the type of the Python object and cast to the appropriate variant type
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
                throw std::runtime_error("Unsupported property type for addVoxelProperty.");
            }

            // Call the C++ function with the converted property value
            return self.addVoxelProperty(property_name, prop_value);
            }, py::arg("property_name"), py::arg("value"))
        .def("setVoxelProperty", 
            [](visioncraft::Model& self, const py::tuple& key_tuple, const std::string& property_name, py::object value) {
                visioncraft::MetaVoxel::PropertyValue prop_value;

                // Check the type of the Python object and assign to the variant accordingly
                if (py::isinstance<py::int_>(value)) {
                    prop_value = value.cast<int>();
                } else if (py::isinstance<py::float_>(value)) {
                    prop_value = value.cast<float>();
                } else if (py::isinstance<py::str>(value)) {
                    prop_value = value.cast<std::string>();
                } else if (py::isinstance<py::array>(value)) {
                    // Handle Eigen::Vector3d explicitly
                    Eigen::Vector3d vec = value.cast<Eigen::Vector3d>();
                    prop_value = vec;
                } else {
                    throw std::runtime_error("Unsupported property type for setMetaVoxelProperty.");
                }

                return self.setVoxelProperty(tupleToKey(key_tuple), property_name, prop_value);
            }, py::arg("key"), py::arg("property_name"), py::arg("value"))
        .def("setMetaVoxelProperty", 
            [](visioncraft::Model& self, const Eigen::Vector3d& position, const std::string& property_name, py::object value) {
                visioncraft::MetaVoxel::PropertyValue prop_value;

                // Check the type of the Python object and assign to the variant accordingly
                if (py::isinstance<py::int_>(value)) {
                    prop_value = value.cast<int>();
                } else if (py::isinstance<py::float_>(value)) {
                    prop_value = value.cast<float>();
                } else if (py::isinstance<py::str>(value)) {
                    prop_value = value.cast<std::string>();
                } else if (py::isinstance<py::array>(value)) {
                    // Handle Eigen::Vector3d explicitly
                    Eigen::Vector3d vec = value.cast<Eigen::Vector3d>();
                    prop_value = vec;
                } else {
                    throw std::runtime_error("Unsupported property type for setMetaVoxelProperty.");
                }

                return self.setVoxelProperty(position, property_name, prop_value);
            }, py::arg("position"), py::arg("property_name"), py::arg("value"))
        .def("getMetaVoxelProperty", [](const visioncraft::Model& self, const py::tuple& key_tuple, const std::string& property_name) -> py::object {
            auto prop = self.getVoxelProperty(tupleToKey(key_tuple), property_name);
            if (auto int_ptr = boost::get<int>(&prop)) {
                return py::int_(*int_ptr);
            } else if (auto float_ptr = boost::get<float>(&prop)) {
                return py::float_(*float_ptr);
            } else if (auto str_ptr = boost::get<std::string>(&prop)) {
                return py::str(*str_ptr);
            } else if (auto vec_ptr = boost::get<Eigen::Vector3d>(&prop)) {
                return py::cast(*vec_ptr);
            } else {
                throw std::runtime_error("Unsupported property type.");
            }
        }, py::arg("key"), py::arg("property_name"))
        .def("getVoxelProperty", [](const visioncraft::Model& self, const Eigen::Vector3d& position, const std::string& property_name) -> py::object {
            auto prop = self.getVoxelProperty(position, property_name);
            if (auto int_ptr = boost::get<int>(&prop)) {
                return py::int_(*int_ptr);
            } else if (auto float_ptr = boost::get<float>(&prop)) {
                return py::float_(*float_ptr);
            } else if (auto str_ptr = boost::get<std::string>(&prop)) {
                return py::str(*str_ptr);
            } else if (auto vec_ptr = boost::get<Eigen::Vector3d>(&prop)) {
                return py::cast(*vec_ptr);
            } else {
                throw std::runtime_error("Unsupported property type.");
            }
        }, py::arg("position"), py::arg("property_name"))
        .def("getVoxelMap", &visioncraft::Model::getVoxelMap, py::return_value_policy::reference);


    // Expose VoxelGridGPU Struct to Python
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
}

#endif // MODEL_BINDINGS_HPP
