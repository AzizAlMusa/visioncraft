#include <pybind11/pybind11.h>

#include "model_bindings.hpp"
#include "viewpoint_bindings.hpp"
#include "visualizer_bindings.hpp"
#include "meta_voxel_bindings.hpp"
#include "meta_voxel_map_bindings.hpp"
#include "visibility_manager_bindings.hpp"


namespace py = pybind11;


PYBIND11_MODULE(visioncraft_py, m) {
    bind_model(m);
    bind_viewpoint(m);
    bind_visualizer(m);
    bind_meta_voxel(m);
    bind_meta_voxel_map(m);
    bind_visibility_manager(m);
}