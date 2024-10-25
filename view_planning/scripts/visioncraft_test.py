import visioncraft_py as vc
import numpy as np
import open3d as o3d

def test_model_loader():
    # Create an instance of ModelLoader
    model_loader = vc.ModelLoader()

    # Load a sample mesh (replace 'path_to_your_mesh_file.ply' with an actual file path)
    print("Loading mesh...")
    success = model_loader.loadModel('../../models/gorilla.ply', 10000, -1)
  
def test_viewpoint():
    # Create an instance of Viewpoint
    viewpoint = vc.Viewpoint()

    # Set the viewpoint's position
    position = np.array([1.0, 2.0, 3.0])
    viewpoint.setPosition(position)
    print("Viewpoint position:", viewpoint.getPosition())

    # Set orientation using a matrix
    orientation_matrix = np.eye(3)  # Identity matrix as orientation
    viewpoint.setOrientation(orientation_matrix)
    print("Viewpoint orientation matrix:\n", viewpoint.getOrientationMatrix())

     # Set orientation using quaternion array (Python array or NumPy array)
    quaternion = np.array([0.707, 0, 0.707, 0])  # Example quaternion
    viewpoint.setOrientation(quaternion)  # Passing a NumPy array of size 4
    print("Viewpoint orientation quaternion:", viewpoint.getOrientationQuaternion())  # Now returns a NumPy array

    # Set lookAt with up vector
    lookAt = np.array([0, 0, 0])  # Look at the origin
    up = np.array([0, 1, 0])  # Up vector in Y-direction
    viewpoint.setLookAt(lookAt, up)
    print("LookAt and orientation set.")

    # Set near and far plane
    viewpoint.setNearPlane(0.1)
    viewpoint.setFarPlane(100.0)
    print("Near plane:", viewpoint.getNearPlane())
    print("Far plane:", viewpoint.getFarPlane())

    # Set resolution
    viewpoint.setResolution(1920, 1080)
    print("Resolution:", viewpoint.getResolution())

    # Set field of view
    viewpoint.setHorizontalFieldOfView(90.0)
    viewpoint.setVerticalFieldOfView(60.0)
    print("Horizontal FOV:", viewpoint.getHorizontalFieldOfView())
    print("Vertical FOV:", viewpoint.getVerticalFieldOfView())

    # Generate rays
    print("Generating rays...")
    rays = viewpoint.generateRays()
    print(f"Generated {len(rays)} rays.")

    # Perform raycasting (assuming you have a valid octomap object)
    # octomap = some_loaded_octomap  # Replace with an actual octomap object
    # print("Performing raycasting...")
    # results = viewpoint.performRaycasting(octomap, use_parallel=True)
    # print("Raycasting complete.")


if __name__ == "__main__":
    print("Testing ModelLoader...")
    # test_model_loader()

    # print("\nTesting Viewpoint...")
    test_viewpoint()
