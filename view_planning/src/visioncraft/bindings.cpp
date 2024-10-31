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
        // Primary constructor with position and orientation
        .def(py::init([](py::array_t<double> position, py::array_t<double> orientation,
                         double near = 350.0, double far = 900.0,
                         int resolution_width = 2448, int resolution_height = 2048,
                         double hfov = 44.8, double vfov = 42.6) {
            // Check shapes for position (1D, shape [3]) and orientation (2D, shape [3, 3])
            if (position.ndim() != 1 || position.shape(0) != 3) {
                throw std::invalid_argument("Position must be a 1D array with shape (3,)");
            }
            if (orientation.ndim() != 2 || orientation.shape(0) != 3 || orientation.shape(1) != 3) {
                throw std::invalid_argument("Orientation must be a 2D array with shape (3, 3)");
            }

            // Copy position and orientation data to Eigen types
            Eigen::Vector3d pos;
            std::memcpy(pos.data(), position.data(), 3 * sizeof(double));
            Eigen::Matrix3d orient;
            std::memcpy(orient.data(), orientation.data(), 9 * sizeof(double));

            // Initialize and return the Viewpoint instance
            return visioncraft::Viewpoint(pos, orient, near, far, resolution_width, resolution_height, hfov, vfov);
        }), py::arg("position"), py::arg("orientation"),
           py::arg("near") = 350.0, py::arg("far") = 900.0,
           py::arg("resolution_width") = 2448, py::arg("resolution_height") = 2048,
           py::arg("hfov") = 44.8, py::arg("vfov") = 42.6)

        // Second Constructor with combined position and Euler angles
        .def(py::init([](py::array_t<double> position_yaw_pitch_roll,
                        double near = 350.0, double far = 900.0,
                        int resolution_width = 2448, int resolution_height = 2048,
                        double hfov = 44.8, double vfov = 42.6) {
            if (position_yaw_pitch_roll.ndim() != 1 || position_yaw_pitch_roll.size() != 6) {
                throw std::invalid_argument("Position with yaw, pitch, and roll must be a 1D array with 6 elements.");
            }

            Eigen::VectorXd pos_ypr(6);
            std::memcpy(pos_ypr.data(), position_yaw_pitch_roll.data(), 6 * sizeof(double));
            
            return visioncraft::Viewpoint(pos_ypr, near, far, resolution_width, resolution_height, hfov, vfov);
        }), py::arg("position_yaw_pitch_roll"),
            py::arg("near") = 350.0, py::arg("far") = 900.0,
            py::arg("resolution_width") = 2448, py::arg("resolution_height") = 2048,
            py::arg("hfov") = 44.8, py::arg("vfov") = 42.6)
        
        // Class method to create with Position and Euler Angles
        .def_static("from_euler", [](py::array_t<double> position_yaw_pitch_roll,
                                                double near = 350.0, double far = 900.0,
                                                int resolution_width = 2448, int resolution_height = 2048,
                                                double hfov = 44.8, double vfov = 42.6) {
                if (position_yaw_pitch_roll.ndim() != 1 || position_yaw_pitch_roll.size() != 6) {
                    throw std::invalid_argument("Position with yaw, pitch, and roll must be a 1D array with 6 elements.");
                }
                Eigen::VectorXd pos_ypr(6);
                std::memcpy(pos_ypr.data(), position_yaw_pitch_roll.data(), 6 * sizeof(double));
                
                // Convert Euler angles to orientation matrix
                Eigen::Vector3d pos = pos_ypr.head<3>();
                Eigen::Vector3d euler = pos_ypr.tail<3>();
                Eigen::Matrix3d orientation = (Eigen::AngleAxisd(euler[2], Eigen::Vector3d::UnitZ()) *
                                Eigen::AngleAxisd(euler[1], Eigen::Vector3d::UnitY()) *
                                Eigen::AngleAxisd(euler[0], Eigen::Vector3d::UnitX()))
                                .toRotationMatrix();

                
                return visioncraft::Viewpoint(pos, orientation, near, far, resolution_width, resolution_height, hfov, vfov);
            }, py::arg("position_yaw_pitch_roll"), py::arg("near") = 350.0, py::arg("far") = 900.0,
            py::arg("resolution_width") = 2448, py::arg("resolution_height") = 2048,
            py::arg("hfov") = 44.8, py::arg("vfov") = 42.6)

        .def_static("from_lookat",
            [](py::array_t<double> position, py::array_t<double> lookAt, 
            py::array_t<double> up = py::none(),
            double near = 350.0, double far = 900.0,
            int resolution_width = 2448, int resolution_height = 2048,
            double hfov = 44.8, double vfov = 42.6) {

                // Validate shapes of position and lookAt arrays
                if (position.ndim() != 1 || position.shape(0) != 3) {
                    throw std::invalid_argument("Position must be a 1D array with shape (3,)");
                }
                if (lookAt.ndim() != 1 || lookAt.shape(0) != 3) {
                    throw std::invalid_argument("LookAt must be a 1D array with shape (3,)");
                }

                // Initialize default up vector to -UnitZ if not provided
                Eigen::Vector3d up_vec = -Eigen::Vector3d::UnitZ();
                if (!up.is_none() && up.ndim() == 1 && up.shape(0) == 3) {
                    std::memcpy(up_vec.data(), up.data(), 3 * sizeof(double));
                }

                // Copy data from numpy arrays to Eigen vectors
                Eigen::Vector3d pos, look_at;
                std::memcpy(pos.data(), position.data(), 3 * sizeof(double));
                std::memcpy(look_at.data(), lookAt.data(), 3 * sizeof(double));

                // Construct and return the Viewpoint object
                return visioncraft::Viewpoint(pos, look_at, up_vec, near, far, resolution_width, resolution_height, hfov, vfov);
            },
            py::arg("position"), py::arg("lookAt"), py::arg("up") = py::none(),
            py::arg("near") = 350.0, py::arg("far") = 900.0,
            py::arg("resolution_width") = 2448, py::arg("resolution_height") = 2048,
            py::arg("hfov") = 44.8, py::arg("vfov") = 42.6)




        .def("setPosition", &visioncraft::Viewpoint::setPosition)
        .def("getPosition", &visioncraft::Viewpoint::getPosition)
        .def("setOrientation", py::overload_cast<const Eigen::Matrix3d&>(&visioncraft::Viewpoint::setOrientation))
        .def("setLookAt", &visioncraft::Viewpoint::setLookAt)
        .def("getOrientationMatrix", &visioncraft::Viewpoint::getOrientationMatrix)
        .def("getOrientationQuaternion", [](const visioncraft::Viewpoint& self) {
            Eigen::Quaterniond quat = self.getOrientationQuaternion();
            py::array_t<double> result({4});
            auto r = result.mutable_unchecked<1>();
            r(0) = quat.w();
            r(1) = quat.x();
            r(2) = quat.y();
            r(3) = quat.z();
            return result;
        })
        // Additional methods for Viewpoint
        .def("setPosition", &visioncraft::Viewpoint::setPosition)
        .def("getPosition", &visioncraft::Viewpoint::getPosition)
        .def("setOrientation", py::overload_cast<const Eigen::Matrix3d&>(&visioncraft::Viewpoint::setOrientation))
        .def("setLookAt", &visioncraft::Viewpoint::setLookAt)
        .def("getOrientationMatrix", &visioncraft::Viewpoint::getOrientationMatrix)
        .def("getOrientationQuaternion", [](const visioncraft::Viewpoint& self) {
            Eigen::Quaterniond quat = self.getOrientationQuaternion();
            py::array_t<double> result({4});
            auto r = result.mutable_unchecked<1>();
            r(0) = quat.w();
            r(1) = quat.x();
            r(2) = quat.y();
            r(3) = quat.z();
            return result;
        })
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
        .def("performRaycastingOnGPU", [](visioncraft::Viewpoint& self, const visioncraft::Model& model) {
            auto hits_map = self.performRaycastingOnGPU(model);

            // Convert the unordered_map of OcTreeKeys to a Python dictionary
            py::dict hits_dict;
            for (auto it = hits_map.begin(); it != hits_map.end(); ++it) {
                hits_dict[keyToTuple(it->first)] = it->second;
            }
            return hits_dict;
        }, py::arg("model"));


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

