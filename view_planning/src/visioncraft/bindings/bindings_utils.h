// octomap_key_helpers.h
#ifndef OCTOMAP_KEY_HELPERS_H
#define OCTOMAP_KEY_HELPERS_H

#include <pybind11/pybind11.h>
#include <octomap/OcTreeKey.h>

namespace py = pybind11;

// Function to convert Python tuple to octomap::OcTreeKey
inline octomap::OcTreeKey tupleToKey(const py::tuple& tuple) {
    if (tuple.size() != 3) {
        throw std::runtime_error("Expected tuple of size 3 for OcTreeKey.");
    }
    return octomap::OcTreeKey(tuple[0].cast<unsigned int>(),
                              tuple[1].cast<unsigned int>(),
                              tuple[2].cast<unsigned int>());
}

// Function to convert octomap::OcTreeKey to Python tuple
inline py::tuple keyToTuple(const octomap::OcTreeKey& key) {
    return py::make_tuple(key.k[0], key.k[1], key.k[2]);
}

#endif // OCTOMAP_KEY_HELPERS_H
