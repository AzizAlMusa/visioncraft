import numpy as np
import time
from visioncraft_py import ModelLoader, Viewpoint, Visualizer

def main():
    # Initialize the ModelLoader and load the model
    model_loader = ModelLoader()
    model_loader.loadModel("../../models/cube.ply", 50000, -1)  # Replace with your actual model path

    # Initialize the Visualizer
    visualizer = Visualizer()

    # Initialize the render window (this is crucial to ensure the render window pops up)
    visualizer.initializeWindow("3D Visualization")
    visualizer.setBackgroundColor(np.array([0.0, 0.0, 0.0]))

    # Create multiple viewpoints
    positions = [
        np.array([400, 0, 0]),  # +X axis
        # np.array([-400, 0, 0]),  # -X axis
    ]

    look_at = np.array([0.0, 0.0, 0.0])  # All viewpoints will look at the origin

    # Perform raycasting for each viewpoint
    for position in positions:
        # Initialize viewpoint with position and look_at
        viewpoint = Viewpoint()
        viewpoint.setPosition(position)
        viewpoint.setLookAt(look_at, np.array([0, -1, 0]))
        viewpoint.setDownsampleFactor(8)  # Set the downsampling factor

        visualizer.addViewpoint(viewpoint)

        # Perform GPU raycasting and measure the execution time
        start_time = time.time()
        hit_results = viewpoint.performRaycastingOnGPU(model_loader)  # Assuming this returns a list of hits
        end_time = time.time()

        # Calculate elapsed time in milliseconds
        elapsed_time = (end_time - start_time) * 1000
        print(f"Raycasting for viewpoint at position {position} took: {elapsed_time:.2f} ms")

        model_loader.updateOctomapWithHits(hit_results)

    # Add octomap with default color (-1 means automatic color from octomap data)
    start_time = time.time()  # Start timer for addOctomap
    visualizer.addOctomap(model_loader)
    end_time = time.time()  # End timer for addOctomap

    elapsed_time = (end_time - start_time) * 1000
    print(f"Adding octomap took: {elapsed_time:.2f} ms")

    # Show the GPU voxel grid
    start_time = time.time()  # Start timer for showGPUVoxelGrid
    # visualizer.showGPUVoxelGrid(model_loader)
    end_time = time.time()  # End timer for showGPUVoxelGrid

    elapsed_time = (end_time - start_time) * 1000
    print(f"Displaying GPU voxel grid took: {elapsed_time:.2f} ms")
    
    # Render the visualization (this starts the render loop)
    visualizer.render()

if __name__ == "__main__":
    main()
