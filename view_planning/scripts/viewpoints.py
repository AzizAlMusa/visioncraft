import numpy as np
import open3d as o3d
from open3d.cuda.pybind.visualization.rendering import MaterialRecord
import matplotlib.pyplot as plt

import pdb

def lookAt(position, target, up):
    forward = np.array(target) - np.array(position)
    forward_norm = np.linalg.norm(forward)

    if forward_norm < 1e-6:
        return np.eye(3)  # Identity matrix as no rotation is needed

    forward /= forward_norm
    
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)

    if right_norm < 1e-6:
        # Re-calculate right using a different axis
        axis = np.array([1.0, 0.0, 0.0]) if abs(forward[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, axis)
        right /= np.linalg.norm(right)

    new_up = np.cross(right, forward)
    
    # Build the 3x3 rotation matrix
    rotation_matrix = np.zeros((3, 3))
    rotation_matrix[:, 0] = right
    rotation_matrix[:, 1] = new_up
    rotation_matrix[:, 2] = forward  # Negative because we're constructing a view matrix

    return rotation_matrix

def get_lookAt_target(camera_pos, rotation_matrix, up_vector):
    # Calculate the forward vector from the rotation matrix
    forward = rotation_matrix[:, 2]

    # Calculate the target position using the camera position and forward vector
    target_pos = camera_pos + forward

    return target_pos

# Modify the Viewpoint class to include the principal point
class Viewpoint:
    def __init__(self, position, orientation, frustum, principal_point, near, far):
        self.position = position
        self.orientation = orientation
        self.frustum = frustum
        self.principal_point = principal_point
        self.intrinsic = None
        self.near = near
        self.far = far

        self.material = MaterialRecord()  # Add this line to initialize the material
        self.material.base_color = [0.8, 0.8, 0.8, 1.0]
        self.material.shader = "defaultLit"


    def initialize_camera_intrinsic(self, width, height, fov_deg):
        fov_rad = np.radians(fov_deg)
        f = width / (2 * np.tan(fov_rad / 2))  # Focal length
        # The principal point is typically at the center of the image
       
        cx = width / 2.0
        cy = height / 2.0
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, f, f, cx, cy)


    def get_extrinsic_matrix(self):
        
        extrinsic_matrix = np.eye(4)  # Initialize a 4x4 identity matrix

        # Assign the rotation matrix to the top-left 3x3 submatrix of the extrinsic matrix
        extrinsic_matrix[:3, :3] = self.orientation

        # Assign the position to the rightmost column of the extrinsic matrix
        extrinsic_matrix[:3, 3] = self.position

        return extrinsic_matrix

    def capture_depth_image(self, mesh, projection_type='intrinsic'):
        width = self.intrinsic.width
        height = self.intrinsic.height

        # Create the OffscreenRenderer
        renderer_pc = o3d.visualization.rendering.OffscreenRenderer(width, height)
        renderer_pc.scene.set_background(np.array([0.5, 0.5, 0.5, 1]))
        renderer_pc.scene.add_geometry("mesh", mesh, self.material)


        renderer_pc.setup_camera(self.intrinsic, self.get_extrinsic_matrix() )
        # Set up Camera for off screen renderer to match viewpoint
        intrinsic_matrix = self.intrinsic.intrinsic_matrix
        renderer_pc.scene.camera.set_projection(intrinsic_matrix, self.near, self.far, 510, 500)    
        if projection_type == 'intrinsic':
            intrinsic_matrix = self.intrinsic.intrinsic_matrix
            renderer_pc.scene.camera.set_projection(intrinsic_matrix, self.near, self.far, 510, 500)
        elif projection_type == 'fov':
            fov_deg = np.degrees(2 * np.arctan(self.intrinsic.width / (2 * self.intrinsic.get_focal_length()[0])))
            aspect_ratio = self.intrinsic.width / self.intrinsic.height
            renderer_pc.scene.camera.set_projection(fov_deg, aspect_ratio, self.near, self.far, renderer_pc.scene.camera.FovType.Vertical)

        

        extrinsic = self.get_extrinsic_matrix()
        cam_position = extrinsic[:3, 3]
        forward = extrinsic[:3, 2]  # Negating because we want to look "into" the camera direction
        target = cam_position + forward  # Or just use [0,0,0] if the camera is always looking at the origin
        up = [0, 1, 0]  # camera orientation
        renderer_pc.scene.camera.look_at(target, cam_position, up)
        # renderer_pc.setup_camera(60.0, target, cam_position, up)
 

        pdb.set_trace()
         # Print matrices
        print("Projection Matrix:")
        print(renderer_pc.scene.camera.get_projection_matrix())
        print("View Matrix:")
        print(renderer_pc.scene.camera.get_view_matrix())
        print("Model Matrix:")
        print(renderer_pc.scene.camera.get_model_matrix())
        # renderer_pc.setup_camera(self.intrinsic, self.get_extrinsic_matrix())
        # Get the depth information
        depth_data = renderer_pc.render_to_depth_image()
        image_data = renderer_pc.render_to_image()

        # depth_data2 = renderer_pc.capture_depth_float_buffer(True)

        # Convert image and depth data to arrays
        image_array = np.asarray(image_data)
        depth_image = np.asarray(depth_data)

        # Normalize the depth image
        normalized_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min() + 1e-10)
        # pdb.set_trace()
        # Create a subplot with 1 row and 2 columns
        plt.subplot(1, 2, 1)
        plt.imshow(image_array)
        plt.axis('off')  # Optional: Turn off axis labels and ticks
        plt.title('Regular Image')

        plt.subplot(1, 2, 2)
        plt.imshow(normalized_image, cmap='gray')
        plt.axis('off')  # Optional: Turn off axis labels and ticks
        plt.title('Depth Image')

        plt.show()

        # Save the depth image
        plt.imsave('depth.png', normalized_image, cmap='gray')

        return depth_data

    def depth_to_pointcloud(self, depth_data):
        # Step 2: Convert depth data to Open3D Image
        
        # Let's say your near plane is at 0.1m and far plane is at 10m
        near_plane = self.near
        far_plane = self.far
        depth_range = far_plane - near_plane

        # Convert normalized depth to actual depth
        depth_array = np.asarray(depth_data)
        actual_depth = depth_array * depth_range  + near_plane
        pdb.set_trace()
        point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
                    o3d.geometry.Image(actual_depth),
                    intrinsic=self.intrinsic,
                    extrinsic=np.linalg.inv(self.get_extrinsic_matrix()),  # This might depend on how your extrinsic matrix is defined.
                    )
        return point_cloud
    

def create_frustum_points(fov_deg, aspect_ratio, near, far):
    half_fov_rad = np.radians(fov_deg / 2)
    half_height_near = np.tan(half_fov_rad) * near
    half_width_near = half_height_near * aspect_ratio

    half_height_far = np.tan(half_fov_rad) * far
    half_width_far = half_height_far * aspect_ratio

    # Near plane points (bottom-left, bottom-right, top-right, top-left)
    nbl = [-half_width_near, -half_height_near, near]
    nbr = [half_width_near, -half_height_near, near]
    ntr = [half_width_near, half_height_near, near]
    ntl = [-half_width_near, half_height_near, near]

    # Far plane points (bottom-left, bottom-right, top-right, top-left)
    fbl = [-half_width_far, -half_height_far, far]
    fbr = [half_width_far, -half_height_far, far]
    ftr = [half_width_far, half_height_far, far]
    ftl = [-half_width_far, half_height_far, far]

    return [nbl, nbr, ntr, ntl, fbl, fbr, ftr, ftl]

def create_viewpoint(position, orientation, fov_deg, aspect_ratio, near=0.1, far=1.0):
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = orientation
    transformation_matrix[:3, 3] = position

    frustum_points = create_frustum_points(fov_deg, aspect_ratio, near, far)
    frustum_lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Near plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Far plane
        [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting lines
    ]

    homogeneous_frustum_points = np.hstack([frustum_points, np.ones((len(frustum_points), 1))])
    transformed_frustum_points = (transformation_matrix @ homogeneous_frustum_points.T).T[:, :3]

    frustum = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(transformed_frustum_points),
        lines=o3d.utility.Vector2iVector(frustum_lines)
    )

    
    # Create a point cloud for the principal point
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector([position])
    
    # Color the principal point red
    pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # RGB
    
    return Viewpoint(position, orientation, frustum, pcd, near, far)



# Load the mesh
mesh = o3d.io.read_triangle_mesh("../models/cube.stl")

# Camera position and orientation
# camera_position = np.array([0, 0, -0.75])
camera_position = np.array([0.0, 0.75, 0.0])
# camera_position = np.array([0.4, 0.4, 0.4])
# rotation_matrix = np.eye(3)
rotation_matrix = lookAt(camera_position, np.array([0, 0, 0]), np.array([0, -1, 0]))
# Field of View in degrees and aspect ratio (e.g., 16/9 for widescreen)
fov_deg = 60
resolution_width, resolution_height = 510, 500 # For example
aspect_ratio = resolution_width / resolution_height

# Create viewpoint (same as before)
viewpoint = create_viewpoint(camera_position, rotation_matrix, fov_deg, aspect_ratio, near=0.1, far=1.0)
viewpoint.initialize_camera_intrinsic(resolution_width, resolution_height, fov_deg)  # Uncomment this line when you add this method to your class


## I WOULD LIKE TO CHECK THE POINT CLOUD FROM THE DEPTH IMAGE HERE
# Assume you already have the depth_image from your capture_depth_image function

# Capture depth image and print matrices with intrinsic-based projection
depth_image = viewpoint.capture_depth_image(mesh, projection_type='intrinsic')


point_cloud = viewpoint.depth_to_pointcloud(depth_image)


# Combine all Open3D-compatible geometries for visualization
geometries = [mesh, viewpoint.frustum, viewpoint.principal_point, point_cloud]

# Compute normals for the mesh
mesh.compute_vertex_normals()


# Set mesh color (RGB)
mesh.paint_uniform_color([0.5, 0.5, 0.5])  # Gray color

# Initialize the visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add geometries to the scene
for geom in geometries:
    vis.add_geometry(geom)

# Customize rendering options
render_option = vis.get_render_option()
render_option.background_color = np.array([0.9, 0.9, 0.9])  # White background
render_option.light_on = True
render_option.show_coordinate_frame = True

# Run the visualizer
vis.run()
vis.destroy_window()