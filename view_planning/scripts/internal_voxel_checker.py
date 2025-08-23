import numpy as np
import sys
import os
import time

# Add path to Python bindings
sys.path.append(os.path.abspath("../build/python_bindings"))
from visioncraft_py import Model, Visualizer

def run_internal_classification():
    # Initialize visualizer
    visualizer = Visualizer()
    visualizer.initializeWindow("3D View")
    visualizer.setBackgroundColor(np.array([0.0, 0.0, 0.0]))

    # Load model
    model = Model()
    model.loadModel("../models/mug.ply", 50000)

    # Perform internal classification using mesh normals and raycasting
    t0 = time.time()
    model.findInternalVoxels(200.0)
    t1 = time.time()

    print(f"[TIMING] findInternalVoxels took {t1 - t0:.3f} seconds")

    # Visualize the internal property: red = internal, white = external
    base_color = np.array([0.0, 1.0, 0.0])      # White for False (external)
    internal_color = np.array([1.0, 0.0, 0.0])  # Red for True (internal)
    visualizer.addVoxelMapProperty(model, "internal", base_color, internal_color)

    print("Press Ctrl+C to exit the viewer loop.")
    try:
        while True:
            visualizer.renderStep()
            time.sleep(0.01)  # Cap to ~100 FPS
    except KeyboardInterrupt:
        print("Viewer loop terminated.")

if __name__ == "__main__":
    run_internal_classification()
