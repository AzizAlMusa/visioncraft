import numpy as np

import sys
import os

sys.path.append(os.path.abspath("../build/python_bindings"))
import visioncraft_py as vc

def compare_viewpoints(vp1, vp2):
    """
    Compare the position and orientation of two viewpoints to check if they are equivalent.
    """
    pos_diff = np.allclose(vp1.getPosition(), vp2.getPosition(), atol=1e-6)
    orient_diff = np.allclose(vp1.getOrientationMatrix(), vp2.getOrientationMatrix(), atol=1e-6)
    return pos_diff and orient_diff

# Define common parameters
position = np.array([400, 0, 0])
orientation_matrix = np.eye(3)  # Example orientation as an identity matrix
euler_angles = np.array([0.0, 0.0, 0.0])  # Yaw, Pitch, Roll as zeros
look_at = np.array([0.0, 0.0, 0.0])
up_vector = np.array([0.0, -1.0, 0.0])

# Initialize viewpoints using each available method
vp_main_constructor = vc.Viewpoint(position, orientation_matrix)
vp_from_euler = vc.Viewpoint.from_position_and_euler(np.hstack([position, euler_angles]))
vp_from_lookat = vc.Viewpoint.from_position_lookat(position, look_at, up_vector)

# Compare all viewpoints with each other
all_comparisons = [
    ("Main constructor vs. from_euler", compare_viewpoints(vp_main_constructor, vp_from_euler)),
    ("Main constructor vs. from_lookat", compare_viewpoints(vp_main_constructor, vp_from_lookat)),
    ("from_euler vs. from_lookat", compare_viewpoints(vp_from_euler, vp_from_lookat)),
]

# Print results
for name, result in all_comparisons:
    print(f"{name}: {'Match' if result else 'Do not match'}")

# Print positions and orientations to verify values
print("\nPositions and Orientations:")
print("Main Constructor Position:", vp_main_constructor.getPosition())
print("Main Constructor Orientation:\n", vp_main_constructor.getOrientationMatrix())

print("\nfrom_euler Position:", vp_from_euler.getPosition())
print("from_euler Orientation:\n", vp_from_euler.getOrientationMatrix())

print("\nfrom_lookat Position:", vp_from_lookat.getPosition())
print("from_lookat Orientation:\n", vp_from_lookat.getOrientationMatrix())