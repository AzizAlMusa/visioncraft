import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Wedge, Circle
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import time
from scipy.interpolate import interp1d

# Configuration parameters
grid_size = 100  # Grid resolution for potential field calculation
frames = 500     # Number of animation frames
num_particles = 9  # Initial number of viewpoints

# Circle parameters
circle_radius = 30  # Radius of the bounding circle for viewpoints
circle_center = np.array([grid_size/2, grid_size/2])  # Center of the circle

# Field of view parameters
fov_angle = 60  # Field of view angle in degrees
fov_radius = 20  # Maximum distance a viewpoint can see

# Visibility threshold for step function
visibility_threshold = 0.5  # Threshold for considering a point "seen"

# Object initialization
theta = np.linspace(0, 2*np.pi, 10, endpoint=False)
radii = 20 + 3*np.random.uniform(-1, 1, size=10)  # random but limited
x = 50 + radii * np.cos(theta)
y = 50 + radii * np.sin(theta)
object_vertices = np.column_stack((x, y))

# # Parameters
# radius = 20
# num_vertices = 20

# # Generate theta values
# theta = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)

# # Calculate object vertices for a fixed radius
# x = 50 + radius * np.cos(theta)
# y = 50 + radius * np.sin(theta)

# # Combine into a single array
object_vertices = np.column_stack((x, y))

# Small value to prevent division by zero
epsilon = 1e-6

class AngularViewPlanningField:
    def __init__(self, grid_size, num_particles, fov_angle, fov_radius, visibility_threshold=1.0):
        self.grid_size = grid_size
        self.fov_angle = fov_angle  # in degrees
        self.fov_radius = fov_radius
        self.visibility_threshold = visibility_threshold
        
        # Object with vertices
        self.object_points = None
        self.obj_visibility = None
        
        # Visibility matrix to track which viewpoint sees which object point
        self.visibility_matrix = None
        
        # For manifold representation
        self.obj_manifold_projections = None  # Angular positions of points projected on manifold
        self.obj_potentials = None  # Potential value for each projected point


        self.visibility_thresholds = [0.95, 1.95, 3.0]  # thresholds at 1, 2, and 3
        self.visibility_outputs = [0.2, 0.0, 1.0, 1.0]  # outputs for each region
        self.visibility_sharpness = 20.0  # sigmoid sharpness

        
    def set_object_points(self, vertices, num_points_per_edge=10):
        """
        Convert object vertices to a set of points along the perimeter.
        """
        object_points = []
        
        # For each edge of the polygon
        for i in range(len(vertices)):
            start = vertices[i]
            end = vertices[(i + 1) % len(vertices)]
            
            # Generate points along this edge
            for t in np.linspace(0, 1, num_points_per_edge):
                point = start + t * (end - start)
                object_points.append(point)
        
        self.object_points = np.array(object_points)
        self.obj_visibility = np.zeros(len(self.object_points))
        
        # Initialize visibility matrix
        # Initially empty since we don't know the number of viewpoints yet
        self.visibility_matrix = None
        
        # Pre-compute angular projections of object points onto the manifold
        self.compute_manifold_projections()
        
        return self.object_points
    
    def compute_manifold_projections(self):
        """
        Project object points onto the manifold (circle) using angular coordinates.
        Each object point projects directly to the manifold.
        """
        if self.object_points is None:
            return
            
        # Calculate vectors from circle center to each object point
        obj_vectors = self.object_points - circle_center
        
        # Convert to angular positions on the manifold
        self.obj_manifold_projections = np.arctan2(obj_vectors[:, 1], obj_vectors[:, 0])
        self.obj_manifold_projections = np.mod(self.obj_manifold_projections, 2*np.pi)  # Ensure [0, 2π)
        
        # Initialize potentials array
        self.obj_potentials = np.zeros(len(self.object_points))
    
    def initialize_viewpoints(self, num_viewpoints, radius, center):
        """
        Initialize viewpoints evenly distributed on a circle using angular coordinates.
        Returns both angular positions and corresponding 2D positions with directions.
        """
        # Angular positions (in radians) - using original range for better stability
        theta = np.linspace(0, 0.2, num_viewpoints, endpoint=False)
        
        # Convert to Cartesian coordinates
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        
        # Direction vectors (pointing to center)
        dx = center[0] - x
        dy = center[1] - y
        
        # Normalize direction vectors
        norm = np.sqrt(dx**2 + dy**2) + epsilon
        dx /= norm
        dy /= norm
        
        # Combine position and direction
        viewpoints = np.column_stack([x, y, dx, dy])
        
        # Initialize visibility matrix with the correct dimensions
        self.visibility_matrix = np.zeros((num_viewpoints, len(self.object_points)))
        
        return viewpoints, theta
    
    def can_see_point(self, viewpoint_pos, direction, obj_point):
        """
        Check how well a viewpoint can see an object point using a sharp sigmoid model.
        Returns a value between 0 (not visible) and 1 (fully visible).
        """
        to_obj = obj_point - viewpoint_pos
        distance = np.linalg.norm(to_obj)
        
        # Hard distance cutoff for speed
        if distance > self.fov_radius * 1.05:
            return 0.0
            
        # Distance factor: Very sharp sigmoid at the edge of FOV radius
        distance_sharpness = 1000.0
        distance_factor = 1.0 / (1.0 + np.exp(distance_sharpness * (distance/self.fov_radius - 1.0)))
        
        # Angle calculation
        to_obj_normalized = to_obj / (distance + epsilon)
        cos_angle = np.dot(to_obj_normalized, direction)
        
        # Convert cos(angle) to angle in radians
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        # Angle factor: Very sharp sigmoid at the edge of FOV angle
        angle_threshold = np.radians(self.fov_angle / 2)
        angle_sharpness = 1000.0
        angle_factor = 1.0 / (1.0 + np.exp(angle_sharpness * (angle/angle_threshold - 1.0)))
        
        # Combine factors - multiply to get overall visibility
        visibility = distance_factor * angle_factor
        
        # Higher threshold for numerical stability to make FOV even sharper
        if visibility < 1e-9:
            return 0.0
            
        return visibility
        
    def update_visibility(self, viewpoints):
        """
        Update visibility matrix and object visibility.
        Returns the coverage (percentage of visible points).
        """
        if self.object_points is None:
            return 0.0
            
        # Reset visibility
        self.obj_visibility.fill(0)
        
        # Extract positions and directions
        positions = viewpoints[:, :2]
        directions = viewpoints[:, 2:]
        
        # Create or resize visibility matrix if needed
        if self.visibility_matrix is None or self.visibility_matrix.shape[0] != len(positions):
            self.visibility_matrix = np.zeros((len(positions), len(self.object_points)))
        else:
            # Reset visibility matrix
            self.visibility_matrix.fill(0)
        
        # For each object point
        for i, obj_point in enumerate(self.object_points):
            for j, (pos, direction) in enumerate(zip(positions, directions)):
                visibility = self.can_see_point(pos, direction, obj_point)
                self.visibility_matrix[j, i] = visibility

        # Calculate the sum of visibility for each object point
        self.obj_visibility = np.sum(self.visibility_matrix, axis=0)
                
        # Calculate visibility function result
        visibility_func_result = self.compute_visibility_function()
        
        # Calculate and return coverage (percentage of visible points)
        # print()
        # coverage = np.sum(visibility_func_result) / len(visibility_func_result)
        coverage = np.sum(self.obj_visibility >= 1) / len(self.obj_visibility) 

        return coverage
    
    # def compute_visibility_function(self):
    #     """
    #     Compute visibility function with a soft threshold using a sigmoid.
    #     Returns an array with values between 0 and 1.
    #     """
    #     # Sigmoid function: 1/(1 + exp(-k*(x-threshold)))
    #     # k controls the steepness of the transition
    #     k = 100.0  # Higher values make the transition sharper
    #     return 1.0 / (1.0 + np.exp(-k * (self.obj_visibility - self.visibility_threshold)))
    
    def compute_visibility_function(self):
        """
        Multi-interval visibility function based on thresholds and outputs.
        """
        x = self.obj_visibility

        t = self.visibility_thresholds
        o = self.visibility_outputs
        k = self.visibility_sharpness

        # Sigmoid transitions
        s = [1.0 / (1.0 + np.exp(-k * (x - threshold))) for threshold in t]

        # Combine intervals
        visibility_output = (
            o[0] * (1 - s[0]) +
            o[1] * (s[0] - s[1]) +
            o[2] * (s[1] - s[2]) +
            o[3] * s[2]
        )

        return np.clip(visibility_output, 0.0, 1.0)


    def compute_potential(self, viewpoints):
        """
        Compute potential for each object point projected onto the manifold.
        Potential = (1 - visibility) * sum(log(distance to each viewpoint))
        """
        # Get visibility function output (1 = seen, 0 = not seen)
        visibility_func_output = self.compute_visibility_function()
        
        # Extract positions for distance calculations
        positions = viewpoints[:, :2]
        
        # For each object point
        for i, obj_point in enumerate(self.object_points):
            # Calculate distances to all viewpoints
            diffs = positions - obj_point
            distances = np.linalg.norm(diffs, axis=1) + epsilon
            
            # Logarithmic potential
            log_distance_sum = np.sum(np.log(distances))
            
            # Potential is (1-visibility_function) * log_distance_sum
            # This means only unseen points have non-zero potential
            self.obj_potentials[i] = (1.0 - visibility_func_output[i]) * log_distance_sum
    
    def compute_angular_forces(self, angular_positions, k_attr=1.0, k_rep=0.2):
        """
        Compute angular forces directly from object points projected onto the manifold:
        - Attraction from low-visibility object points
        - Repulsion between viewpoints
        """
        num_viewpoints = len(angular_positions)
        angular_forces = np.zeros(num_viewpoints)

        angular_attractive_forces = np.zeros(num_viewpoints)
        angular_repulsive_forces = np.zeros(num_viewpoints)

        # --- Attractive Force from Object Points with Potential ---
        for j in range(num_viewpoints):
            theta_j = angular_positions[j]
            force = 0.0
            
            # Calculate force from each object point based on its potential
            for i in range(len(self.obj_manifold_projections)):
                # Only consider points with potential above a small threshold
                if np.abs(self.obj_potentials[i]) > 0.0:  # Small threshold for numerical stability
                    theta_i = self.obj_manifold_projections[i]
                    
                    # Calculate angular difference (shortest path on circle)
                    delta_theta = theta_i - theta_j
                    delta_theta = np.arctan2(np.sin(delta_theta), np.cos(delta_theta))
                    
                    distance = np.abs(delta_theta) + epsilon
                    sign = np.sign(delta_theta)
                    
                    # Force is proportional to potential and inversely to distance
                    # Using the object's potential directly here is the key
                    distance = max(np.abs(distance), np.pi/180 * 1)
                    force += self.obj_potentials[i] * (1.0 / distance) * sign
            
            angular_attractive_forces[j] = k_attr * force
        # --- Repulsive Force Between Viewpoints ---
        if k_rep > 0 and num_viewpoints > 1:
            sigma = 1.5 # Width of Gaussian repulsion
            for i in range(num_viewpoints):
                for j in range(num_viewpoints):
                    if i != j:
                        delta_theta = angular_positions[j] - angular_positions[i]
                        delta_theta = np.arctan2(np.sin(delta_theta), np.cos(delta_theta))
                        
                        # Gaussian repulsion
                        gaussian_repulsion = -np.sign(delta_theta) * k_rep * np.exp(-(delta_theta**2) / (2*sigma**2))
                        
                        angular_repulsive_forces[i] += gaussian_repulsion
                 
        print(angular_attractive_forces)
        print(angular_repulsive_forces)
        print("==================")
        angular_forces = angular_attractive_forces + angular_repulsive_forces

        return angular_forces
    
    def update_viewpoints(self, viewpoints, angular_positions, learning_rate=0.01,
                      m=None, v=None, t=0, beta1=0.9, beta2=0.999, eps=1e-8, 
                      k_attr=0.4, k_rep=0.2):
        """
        Update viewpoint positions using angular coordinates with Adam optimizer.
        """
        # Calculate angular forces directly from projected object points
        angular_forces = self.compute_angular_forces(angular_positions, 
                                               k_attr=k_attr, k_rep=k_rep)
        
        # Apply damping to reduce oscillations
        # if t > 50:  # After initial exploration phase
        #     angular_forces *= 0.9  # Damping factor
        # print(angular_forces)
        # print("==================")
        # Adam optimizer update
        if m is None:
            m = np.zeros_like(angular_positions)
        if v is None:
            v = np.zeros_like(angular_positions)
            
        t += 1
        m = beta1 * m + (1 - beta1) * angular_forces
        v = beta2 * v + (1 - beta2) * (angular_forces ** 2)
        
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        # Update angular positions with fixed learning rate
        angular_positions += learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        
        # Ensure angles stay within [0, 2π)
        angular_positions = angular_positions % (2 * np.pi)
        
        # Convert back to Cartesian coordinates
        x = circle_center[0] + circle_radius * np.cos(angular_positions)
        y = circle_center[1] + circle_radius * np.sin(angular_positions)
        
        # Update positions in viewpoints array
        viewpoints[:, 0] = x
        viewpoints[:, 1] = y
        
        # Update directions to point to center
        dx = circle_center[0] - x
        dy = circle_center[1] - y
        
        # Normalize direction vectors
        norm = np.sqrt(dx**2 + dy**2) + epsilon
        dx /= norm
        dy /= norm
        
        # Update directions
        viewpoints[:, 2] = dx
        viewpoints[:, 3] = dy
        
        return viewpoints, angular_positions, m, v, t

# Initial setup for the field
field = AngularViewPlanningField(grid_size, num_particles, fov_angle, fov_radius, 
                                visibility_threshold=visibility_threshold)
object_points = field.set_object_points(object_vertices, num_points_per_edge=10)
viewpoints, angular_positions = field.initialize_viewpoints(num_particles, circle_radius, circle_center)

# Calculate initial visibility and potential
coverage = field.update_visibility(viewpoints)
field.compute_potential(viewpoints)

# Setup figure for animation
fig = plt.figure(figsize=(18, 6), dpi=100)
gs = GridSpec(2, 3, height_ratios=[2, 1], width_ratios=[1, 1, 1])

# Main visualization area
ax_main = fig.add_subplot(gs[0, 0])
ax_main.set_xlim(0, grid_size)
ax_main.set_ylim(0, grid_size)
ax_main.set_aspect("equal")
ax_main.set_title("Angular View Planning Simulation", fontsize=12)

# Coverage plot
ax_coverage = fig.add_subplot(gs[0, 1])
ax_coverage.set_xlim(0, frames)
ax_coverage.set_ylim(0, 1)
ax_coverage.set_xlabel("Time Step", fontsize=10)
ax_coverage.set_ylabel("Coverage", fontsize=10)
ax_coverage.set_title("Coverage Over Time", fontsize=10)

# Manifold plot - showing projected object points directly
ax_manifold = fig.add_subplot(gs[0, 2])
ax_manifold.set_xlim(0, 2*np.pi)
ax_manifold.set_ylim(0, np.max(field.obj_potentials)*1.1)  # Adjust based on potential values
ax_manifold.set_xlabel("Angular Position (radians)", fontsize=10)
ax_manifold.set_ylabel("Potential", fontsize=10)
ax_manifold.set_title("Manifold Potentials", fontsize=10)
ax_manifold.set_xticks(np.linspace(0, 2*np.pi, 5))
ax_manifold.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

# Step function visualization (in bottom row)
ax_step = fig.add_subplot(gs[1, :])
ax_step.set_xlim(0, max(field.visibility_thresholds) + 1)
ax_step.set_ylim(0, 1.1)
ax_step.set_xlabel("Visibility (Sum over Viewpoints)", fontsize=10)
ax_step.set_ylabel("Step Function Output", fontsize=10)
ax_step.set_title("Multi-Level Visibility Step Function", fontsize=10)

t = field.visibility_thresholds
o = field.visibility_outputs
k = field.visibility_sharpness

# Plot
x_step = np.linspace(0, max(t) + 1, 400)
s = [1.0 / (1.0 + np.exp(-k * (x_step - threshold))) for threshold in t]

y_step = (
    o[0] * (1 - s[0]) +
    o[1] * (s[0] - s[1]) +
    o[2] * (s[1] - s[2]) +
    o[3] * s[2]
)

step_line, = ax_step.plot(x_step, y_step, 'k-', lw=2)

# Add scatter for object points on the step function
step_scatter = ax_step.scatter([], [], c=[], cmap='plasma', s=50, alpha=0.7)


# Initialize visualization
# Adjust spacing
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# Plot elements
# Polygon
polygon = plt.Polygon(object_vertices, fill=True, alpha=0.3, color='gray')
ax_main.add_patch(polygon)

# Bounding circle
circle = plt.Circle(circle_center, circle_radius, fill=False, color='black', linestyle='--')
ax_main.add_patch(circle)

# Simple circle outline for the manifold
manifold_circle = plt.Circle(circle_center, circle_radius, fill=False, color='lightgray', linestyle='-', alpha=0.5)
ax_main.add_patch(manifold_circle)

# Create projected points on the manifold (Set S) with coloring based on potential
projected_x = circle_center[0] + circle_radius * np.cos(field.obj_manifold_projections)
projected_y = circle_center[1] + circle_radius * np.sin(field.obj_manifold_projections)

# Create scatter for projected points S on the manifold
manifold_projected_scatter = ax_main.scatter(
    projected_x, projected_y,
    c=field.obj_potentials,  # Color by potential
    cmap='plasma',
    s=40,  # Slightly larger to make them visible
    alpha=0.7,
    zorder=2,
    edgecolors='white'  # White edges to make them stand out
)

# Object points with visibility coloring
visibility_function_output = field.compute_visibility_function()

# # Assuming visibility_function_output is a list or array of visibility values in range [0, 1]
# visibility_function_output = np.clip(visibility_function_output, 0, 1)  # Clip values between 0 and 1
# # Create a colormap from red to green
# cmap = plt.get_cmap('RdYlGn')  # Red to Green colormap
# # Normalize visibility values between 0 and 1
# norm = mcolors.Normalize(vmin=0, vmax=1)
# # Apply the colormap to map values to colors
# visibility_colors = [cmap(norm(v)) for v in visibility_function_output]

# # visibility_colors will now have RGBA values, where 0 is red and 1 is green

# object_scatter = ax_main.scatter(object_points[:, 0], object_points[:, 1], 
#                                c=visibility_colors, 
#                                s=20, alpha=0.8, zorder=3)

# Assuming visibility_function_output is a list or array of visibility values
visibility_function_output = field.compute_visibility_function()

# Clip values between 0 and None, allowing values greater than 2
visibility_function_output = np.clip(visibility_function_output, 0, None)

# Create a custom function to map values to the desired colors
def custom_color_map(val):
    if val < 1:
        # Interpolate from red (1, 0, 0) to light green (0.5, 1, 0.5) for values between 0 and 1
        return (1, 1 - val, 1 - val)  # Red to light green
    elif val < 2:
        # Interpolate from light green (0.5, 1, 0.5) to green (0, 1, 0) for values between 1 and 2
        return (0.5, 1 - (val - 1) * 0.5, (val - 1) * 0.5)  # Light green to green
    else:
        # Values greater than 2 become dark green
        return (0, 0.5, 0)  # Dark green

# Apply the custom color function
visibility_colors = [custom_color_map(v) for v in visibility_function_output]

# Object points with visibility coloring
object_scatter = ax_main.scatter(object_points[:, 0], object_points[:, 1], 
                                 c=visibility_colors, 
                                 s=20, alpha=0.8, zorder=3)

# Viewpoints
viewpoint_scatter = ax_main.scatter(viewpoints[:, 0], viewpoints[:, 1], 
                                 c='blue', s=50, zorder=4)

# FOV visualization
fov_patches = []
for i in range(viewpoints.shape[0]):
    pos = viewpoints[i, :2]
    direction = viewpoints[i, 2:]
    
    # Calculate angle in degrees
    angle = np.degrees(np.arctan2(direction[1], direction[0]))
    
    # Outer wedge
    wedge = Wedge(pos, fov_radius, angle - fov_angle/2, angle + fov_angle/2, 
                 alpha=0.3, color='blue', zorder=1, width=0)
    ax_main.add_patch(wedge)
    
    # Add border
    edge_wedge = Wedge(pos, fov_radius, angle - fov_angle/2, angle + fov_angle/2, 
                     alpha=0.6, color='blue', zorder=1, width=fov_radius*0.03)
    ax_main.add_patch(edge_wedge)
    
    fov_patches.append(wedge)
    fov_patches.append(edge_wedge)

# Coverage line
coverage_line, = ax_coverage.plot([], [], color='blue')
coverage_data = [coverage]

# Object point projections on the manifold visualization
manifold_object_scatter = ax_manifold.scatter(
    field.obj_manifold_projections,
    field.obj_potentials,
    c=field.obj_potentials,
    cmap='plasma',
    s=50, 
    alpha=0.7, 
    zorder=2
)

# Add viewpoint positions on manifold
manifold_viewpoint_scatter = ax_manifold.scatter(
    angular_positions,
    np.zeros_like(angular_positions),  # At the bottom of the plot
    c='blue', 
    s=80, 
    marker='^', 
    zorder=3
)

# Horizontal line for zero potential reference
ax_manifold.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

# Text for stats
stats_text = ax_main.text(0.02, 0.98, "", transform=ax_main.transAxes, 
                        ha='left', va='top', fontsize=10)

# Add colorbar for potentials
cbar = plt.colorbar(manifold_projected_scatter, ax=ax_main)
cbar.set_label('Object Point Potential')

# Initialize optimizer variables
m_angular = np.zeros_like(angular_positions)
v_angular = np.zeros_like(angular_positions)
t_angular = 0

# Max potential value seen for consistent colormap
max_potential_seen = np.max(field.obj_potentials) if np.max(field.obj_potentials) > 0 else 1.0

def update(frame):
    global viewpoints, angular_positions, m_angular, v_angular, t_angular, fov_patches
    global max_potential_seen
    
    start_time = time.time()
    
    # First, update visibility
    coverage = field.update_visibility(viewpoints)
    coverage_data.append(coverage)
    
    # Get step function output for visualization
    visibility_function_output = field.compute_visibility_function()
    
    # Second, compute potential based on visibility
    field.compute_potential(viewpoints)

    # Update max potential for color scaling
    current_max = np.max(field.obj_potentials)
    if current_max > 0:
        max_potential_seen = max(max_potential_seen, current_max)
    
    # Update viewpoints with fixed learning rate
    viewpoints, angular_positions, m_angular, v_angular, t_angular = field.update_viewpoints(
        viewpoints, angular_positions, learning_rate=0.05, 
        m=m_angular, v=v_angular, t=t_angular,
        k_attr=2.0, k_rep=0.0  # Use the same repulsion coefficient as in the class definition
    )
    
    # Update visualization
    
    # 1. Update object points coloring by visibility
    # Use visibility function output (red for invisible, green for visible) based on threshold
    # visibility_function_output = np.clip(visibility_function_output, 0, 1)  # Clip values between 0 and 1
    # # Create a colormap from red to green
    # cmap = plt.get_cmap('RdYlGn')  # Red to Green colormap
    # # Normalize visibility values between 0 and 1
    # norm = mcolors.Normalize(vmin=0, vmax=1)
    # # Apply the colormap to map values to colors
    # visibility_colors = [cmap(norm(v)) for v in visibility_function_output]
    # object_scatter.set_color(visibility_colors)

    # Assuming self.obj_visibility contains the visibility values
    visibility_function_output = field.obj_visibility

    # Clip values between 0 and None, allowing values greater than 1
    visibility_function_output = np.clip(visibility_function_output, 0, None)

    # Create a custom function to map values to the desired colors
    def custom_color_map(val):
        if val < 1:
            # Interpolate from red (1, 0, 0) to yellow (1, 1, 0) for values between 0 and 0.99
            return (1 , 0, 0)  # Transition from red to yellow
        elif val == 1:
            # Exactly 1 becomes green
            return (0, 1, 0)  # Green
        else:
            # Values greater than 1 become dark green, darkening with increasing value
            dark_green_factor = max(0, 1 - (val - 1) * 0.2)  # Darkens green as value increases
            return (0, dark_green_factor, 0)  # Darker green as the value increases

    # Apply the custom color function
    visibility_colors = [custom_color_map(v) for v in visibility_function_output]

    # Object points with visibility coloring
    object_scatter = ax_main.scatter(object_points[:, 0], object_points[:, 1], 
                                    c=visibility_colors, 
                                    s=20, alpha=0.8, zorder=3)
    
    # 2. Update viewpoint positions
    viewpoint_scatter.set_offsets(viewpoints[:, :2])
    
    # 3. Update FOV wedges
    for wedge in fov_patches:
        wedge.remove()
    
    fov_patches.clear()
    for i in range(viewpoints.shape[0]):
        pos = viewpoints[i, :2]
        direction = viewpoints[i, 2:]
        
        # Calculate angle in degrees
        angle = np.degrees(np.arctan2(direction[1], direction[0]))
        
        # Create wedge with sharper appearance
        wedge = Wedge(pos, fov_radius, angle - fov_angle/2, angle + fov_angle/2, 
                     alpha=0.3, color='blue', zorder=1)
        ax_main.add_patch(wedge)
        
        # Add edge for sharper appearance
        edge_wedge = Wedge(pos, fov_radius, angle - fov_angle/2, angle + fov_angle/2, 
                         alpha=0.6, color='blue', zorder=1, width=fov_radius*0.03)
        ax_main.add_patch(edge_wedge)
        
        fov_patches.append(wedge)
        fov_patches.append(edge_wedge)
    
    # 4. Update coverage line
    coverage_line.set_data(range(len(coverage_data)), coverage_data)
    # Dynamically expand x-axis as needed
    if len(coverage_data) > ax_coverage.get_xlim()[1]:
        ax_coverage.set_xlim(0, len(coverage_data) + 10)  # Add some padding
    
    # 5. Update the projected points on the manifold in the main view
    manifold_projected_scatter.set_offsets(np.column_stack([projected_x, projected_y]))
    manifold_projected_scatter.set_array(field.obj_potentials)
    manifold_projected_scatter.set_clim(vmin=0, vmax=max(max_potential_seen, 0.01))
    
    # 6. Update manifold visualization with object projections in the graph
    manifold_object_scatter.set_offsets(np.column_stack([
        field.obj_manifold_projections, 
        field.obj_potentials
    ]))
    
    # Update colors to match potentials
    manifold_object_scatter.set_array(field.obj_potentials)
    
    # Update color normalization
    manifold_object_scatter.set_clim(vmin=0, vmax=max(max_potential_seen, 0.01))
    ax_manifold.set_ylim(0, max(max_potential_seen*1.1, 0.01))  # Dynamic y-axis scaling
    
    # 7. Update viewpoint positions on manifold
    manifold_viewpoint_scatter.set_offsets(np.column_stack([
        angular_positions, 
        np.zeros_like(angular_positions)
    ]))
    
    # 8. Update step function visualization with current object points
    step_scatter.set_offsets(np.column_stack([
        field.obj_visibility,  # Raw visibility values
        visibility_function_output  # Step function output (0 or 1)
    ]))
    step_scatter.set_array(field.obj_potentials)
    step_scatter.set_clim(vmin=0, vmax=max(max_potential_seen, 0.01))
    
    # 9. Update stats text
    elapsed = time.time() - start_time
    visible_count = np.sum(visibility_function_output)
    total_count = len(visibility_function_output)
    stats_text.set_text(f"Frame: {frame}\n"
                       f"Coverage: {coverage:.2f}\n"
                       f"Visible: {visible_count}/{total_count}\n"
                       f"Time: {elapsed:.3f}s")
    
    return [object_scatter, viewpoint_scatter, manifold_projected_scatter, manifold_object_scatter, 
            manifold_viewpoint_scatter, coverage_line, stats_text, step_scatter]

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=False)

plt.tight_layout()
# ani.save('view_planning_with_visibility_step.mp4', writer='ffmpeg', fps=60, dpi=100)
plt.show()
# ani.save('view_planning_with_step_function.mp4', writer='ffmpeg', fps=120, dpi=100)