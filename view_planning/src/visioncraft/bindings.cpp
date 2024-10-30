#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "visioncraft/model.h"
#include "visioncraft/viewpoint.h"
#include "visioncraft/visualizer.h"
#include "visioncraft/meta_voxel.h"
#include "visioncraft/meta_voxel_map.h"



namespace py = pybind11;


// Helper function to convert Python tuple to octomap::OcTreeKey
octomap::OcTreeKey tupleToKey(const py::tuple& tuple) {
    if (tuple.size() != 3) {
        throw std::runtime_error("Expected tuple of size 3 for OcTreeKey.");
    }
    return octomap::OcTreeKey(tuple[0].cast<unsigned int>(), 
                            tuple[1].cast<unsigned int>(), 
                            tuple[2].cast<unsigned int>());
}

// Helper function to convert octomap::OcTreeKey to Python tuple
py::tuple keyToTuple(const octomap::OcTreeKey& key) {
    return py::make_tuple(key.k[0], key.k[1], key.k[2]);
}


PYBIND11_MODULE(visioncraft_py, m) {


    // Expose the Model class to Python
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
        }, py::arg("position"), py::arg("property_name"));



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
        .def("performRaycasting", [](visioncraft::Viewpoint &self, const visioncraft::Model& model, bool use_parallel) {
            // Call the original performRaycasting method
            auto result = self.performRaycasting(model, use_parallel);

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
            .def("addOctomap", &visioncraft::Visualizer::addOctomap, py::arg("model"), py::arg("color") = Eigen::Vector3d(-1, -1, -1))
            .def("showGPUVoxelGrid", &visioncraft::Visualizer::showGPUVoxelGrid, py::arg("model"), py::arg("color") = Eigen::Vector3d(1, 0, 0))
            .def("setBackgroundColor", &visioncraft::Visualizer::setBackgroundColor)
            .def("setViewpointFrustumColor", &visioncraft::Visualizer::setViewpointFrustumColor)
            .def("render", &visioncraft::Visualizer::render);

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

