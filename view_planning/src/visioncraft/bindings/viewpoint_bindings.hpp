#ifndef VIEWPOINT_BINDINGS_HPP
#define VIEWPOINT_BINDINGS_HPP

#include <pybind11/pybind11.h>

#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "bindings_utils.h"

#include "visioncraft/viewpoint.h"


namespace py = pybind11;

inline void bind_viewpoint(py::module& m) {
    // Expose the Viewpoint class to Python
    py::class_<visioncraft::Viewpoint, std::shared_ptr<visioncraft::Viewpoint>>(m, "Viewpoint")
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
}

#endif // VIEWPOINT_BINDINGS_HPP
