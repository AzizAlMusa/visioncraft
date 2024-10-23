import numpy as np
import time
from visioncraft_py import ModelLoader, Viewpoint

def main():
    # Initialize the ModelLoader and load the model
    model_loader = ModelLoader()
    model_loader.loadModel("../../models/cube.ply", 50000, -1)  # Replace with your actual model path

    # Create multiple viewpoints

    # Create 20 viewpoints of radius 400 around the origin
    positions = []
    for i in range(20):
        angle = i * np.pi / 10
        x = 400 * np.cos(angle)
        y = 400 * np.sin(angle)
        positions.append(np.array([x, y, 0]))


    look_at = np.array([0.0, 0.0, 0.0])  # All viewpoints will look at the origin

    # Perform raycasting for each viewpoint
    for position in positions:
        # Initialize viewpoint with position and look_at
        viewpoint = Viewpoint()
        viewpoint.setPosition(position)
        viewpoint.setLookAt(look_at, np.array([0, -1, 0]))
        viewpoint.setDownsampleFactor(4)  # Set the downsampling factor

        # Perform GPU raycasting and measure the execution time
        start_time = time.time()
        hit_results = viewpoint.performRaycastingOnGPU(model_loader)  # Assuming this returns a list of hits
        end_time = time.time()

        # Calculate elapsed time in milliseconds
        elapsed_time = (end_time - start_time) * 1000
        print(f"Raycasting for viewpoint at position {position} took: {elapsed_time:.2f} ms")

        # Print the number of hits from GPU raycasting
        # print(f"Number of voxels hit by GPU raycasting: {len(hit_results)}")

if __name__ == "__main__":
    main()
