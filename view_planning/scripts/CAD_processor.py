import open3d as o3d
import numpy as np
import argparse

def scale_to_unit_box(mesh, target_size=100.0):
    bbox = mesh.get_axis_aligned_bounding_box()
    scale = target_size / max(bbox.get_extent())
    mesh.scale(scale, center=bbox.get_center())
    return mesh

def convert_stl_to_ply(input_stl, output_ply):
    mesh = o3d.io.read_triangle_mesh(input_stl)
    if not mesh.has_triangles():
        raise RuntimeError("Failed to load STL or mesh is empty.")

    mesh = scale_to_unit_box(mesh, target_size=100.0)
    mesh.compute_vertex_normals()
    
    success = o3d.io.write_triangle_mesh(output_ply, mesh)
    if not success:
        raise RuntimeError("Failed to write PLY file.")
    print(f"Saved scaled mesh to: {output_ply}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale STL to 10x10x10 box and save as PLY.")
    parser.add_argument("input_stl", help="Path to input STL file")
    parser.add_argument("output_ply", help="Path to output PLY file")
    args = parser.parse_args()

    convert_stl_to_ply(args.input_stl, args.output_ply)
