import numpy as np
import sys
import os
import time

# Adjust path to locate bindings if necessary
sys.path.append(os.path.abspath("../build/python_bindings"))
import visioncraft_py as vc

def update_voxel_map_visibility(octree_hits, model):
    """
    Update the visibility property for each hit voxel in the MetaVoxelMap based on octree hits.

    Args:
        octree_hits: A dictionary where keys are voxel positions (tuple) and values are hit status (bool).
        model: The visioncraft Model object containing the MetaVoxelMap.
    """
    for key, hit in octree_hits.items():
        if not hit:
            continue  # Skip if voxel wasn't hit

        meta_voxel = model.getVoxel(key)
        if meta_voxel is not None:
            visibility = meta_voxel.getProperty("visibility") + 1
            meta_voxel.setProperty("visibility", visibility)
        else:
            print(f"No MetaVoxel found for key {key}")

# Initialize the Visualizer
visualizer = vc.Visualizer()
visualizer.initializeWindow("3D View")
visualizer.setBackgroundColor(np.array([0.0, 0.0, 0.0]))

# Load model
model = vc.Model()
model.loadModel("../models/cube.ply", 50000)
model.addVoxelProperty("visibility", 0)


# Define positions
# Define positions at a radius of 400 from the center
radius = 400
positions = [
    np.array([radius, 0.0, 0.0]),   # Position 1: +X direction
    np.array([-radius, 0.0, 0.0]),  # Position 2: -X direction
    np.array([0.0, radius, 0.0]),   # Position 3: +Y direction
    np.array([0.0, -radius, 0.0]),  # Position 4: -Y direction
    np.array([0.0, 0.0, radius]),   # Position 5: +Z direction
    np.array([0.0, 0.0, -radius]),  # Position 6: -Z direction

]

# Continue with the rest of your code...



look_at = np.array([0.0, 0.0, 0.0])



# Perform raycasting and update visibility
for idx, position in enumerate(positions):

    viewpoint = vc.Viewpoint.from_lookat(position, look_at)
    viewpoint.setDownsampleFactor(8)

    print("Orientation Matrix:", viewpoint.getOrientationMatrix())

    # Perform raycasting on GPU
    start_time = time.time()
    hit_results = viewpoint.performRaycastingOnGPU(model)
    elapsed = time.time() - start_time
    print(f"Raycasting for viewpoint {idx + 1} took: {elapsed * 1000:.2f} ms")

    # Update visibility in the MetaVoxelMap based on hit results
    update_voxel_map_visibility(hit_results, model)

    # Add viewpoint to visualizer
    visualizer.addViewpoint(viewpoint, showFrustum=True, showAxes=True)

# Visualize results
baseColor = np.array([1.0, 1.0, 1.0])      # Base color for the voxel map
propertyColor = np.array([0.0, 1.0, 0.0])   # Color to represent visibility
visualizer.addVoxelMapProperty(model, "visibility", baseColor, propertyColor, 0, 2)

# Start rendering loop
visualizer.render()
