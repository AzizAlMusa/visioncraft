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
frames = 100     # Number of animation frames
num_particles = 6  # Initial number of viewpoints

# Circle parameters
circle_radius = 30  # Radius of the bounding circle for viewpoints
circle_center = np.array([grid_size/2, grid_size/2])  # Center of the circle

# Field of view parameters
fov_angle = 60  # Field of view angle in degrees
fov_radius = 20  # Maximum distance a viewpoint can see

# Object initialization - simple circle for now
# theta = np.linspace(0, 2*np.pi, 36, endpoint=False)
# radii = 20 + 3 * np.sin(6 * theta)  # 6 "petals"
# x = 50 + radii * np.cos(theta)
# y = 50 + radii * np.sin(theta)
# object_vertices = np.column_stack((x, y))

# theta = np.linspace(0, 2*np.pi, 12, endpoint=False)
# radii = 18 + 4*np.random.rand(12)  # slight random bumps
# x = 50 + radii * np.cos(theta)
# y = 50 + radii * np.sin(theta)
# object_vertices = np.column_stack((x, y))


theta = np.linspace(0, 2*np.pi, 10, endpoint=False)
radii = 20 + 3*np.random.uniform(-1, 1, size=10)  # random but limited
x = 50 + radii * np.cos(theta)
y = 50 + radii * np.sin(theta)
object_vertices = np.column_stack((x, y))


# Small value to prevent division by zero
epsilon = 1e-6

class AngularViewPlanningField:
    def __init__(self, grid_size, num_particles, fov_angle, fov_radius):
        self.grid_size = grid_size
        self.fov_angle = fov_angle  # in degrees
        self.fov_radius = fov_radius
        
        # Object with vertices
        self.object_points = None
        self.obj_visibility = None
        
        # For manifold representation
        self.obj_manifold_projections = None  # Angular positions of points projected on manifold
        self.obj_potentials = None  # Potential value for each projected point
        
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
        # Angular positions (in radians)
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
        distance_sharpness = 25.0
        distance_factor = 1.0 / (1.0 + np.exp(distance_sharpness * (distance/self.fov_radius - 0.95)))
        
        # Angle calculation
        to_obj_normalized = to_obj / (distance + epsilon)
        cos_angle = np.dot(to_obj_normalized, direction)
        
        # Convert cos(angle) to angle in radians
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        # Angle factor: Very sharp sigmoid at the edge of FOV angle
        angle_threshold = np.radians(self.fov_angle / 2)
        angle_sharpness = 20.0
        angle_factor = 1.0 / (1.0 + np.exp(angle_sharpness * (angle/angle_threshold - 0.95)))
        
        # Combine factors - multiply to get overall visibility
        visibility = distance_factor * angle_factor
        
        # Higher threshold for numerical stability to make FOV even sharper
        if visibility < 0.05:
            return 0.0
            
        return visibility
        
    def update_visibility(self, viewpoints):
        """
        Update visibility of object points only. Separate from potential calculation.
        """
        if self.object_points is None:
            return 0.0
            
        # Reset visibility
        self.obj_visibility.fill(0)
        
        # Extract positions and directions
        positions = viewpoints[:, :2]
        directions = viewpoints[:, 2:]
        
        # For each object point
        for i, obj_point in enumerate(self.object_points):
            # Check visibility from all viewpoints and take the maximum
            max_visibility = 0.0
            for j, (pos, direction) in enumerate(zip(positions, directions)):
                visibility = self.can_see_point(pos, direction, obj_point)
                max_visibility = max(max_visibility, visibility)
            
            # Store the maximum visibility for this point
            self.obj_visibility[i] = max_visibility
        
        # Calculate and return average coverage
        coverage = np.mean(self.obj_visibility)
        return coverage
        
    def compute_potential(self, viewpoints):
        """
        Compute potential for each object point projected onto the manifold.
        Potential = (1 - visibility) * sum(log(distance to each viewpoint))
        """
        # Extract positions for distance calculations
        positions = viewpoints[:, :2]
        
        # For each object point
        for i, obj_point in enumerate(self.object_points):
            # Calculate distances to all viewpoints
            diffs = positions - obj_point
            distances = np.linalg.norm(diffs, axis=1) + epsilon
            
            # Logarithmic potential
            log_distance_sum = np.sum(np.log(distances))
            
            # Potential is (1-visibility) * log_distance_sum
            self.obj_potentials[i] = (1.0 - self.obj_visibility[i]) * log_distance_sum
    
    def compute_angular_forces(self, angular_positions, alpha=1.0, k_rep=0.2):
        """
        Compute angular forces directly from object points projected onto the manifold:
        - Attraction from low-visibility object points
        - Repulsion between viewpoints
        """
        num_viewpoints = len(angular_positions)
        angular_forces = np.zeros(num_viewpoints)

        # --- Attractive Force from Object Points with Potential ---
        for j in range(num_viewpoints):
            theta_j = angular_positions[j]
            force = 0.0
            
            # Calculate force from each object point based on its potential
            for i in range(len(self.obj_manifold_projections)):
                # Only consider points with potential above a small threshold
                if np.abs(self.obj_potentials[i]) > 0.01:  # Small threshold for numerical stability
                    theta_i = self.obj_manifold_projections[i]
                    
                    # Calculate angular difference (shortest path on circle)
                    delta_theta = theta_i - theta_j
                    delta_theta = np.arctan2(np.sin(delta_theta), np.cos(delta_theta))
                    
                    distance = np.abs(delta_theta) + epsilon
                    sign = np.sign(delta_theta)
                    
                    # Force is proportional to potential and inversely to distance
                    # Using the object's potential directly here is the key
                    force += self.obj_potentials[i] * (1.0 / distance) * sign
            
            angular_forces[j] = alpha * force

        # --- Repulsive Force Between Viewpoints ---
        if k_rep > 0 and num_viewpoints > 1:
            sigma = 0.1  # Width of Gaussian repulsion
            for i in range(num_viewpoints):
                for j in range(num_viewpoints):
                    if i != j:
                        delta_theta = angular_positions[j] - angular_positions[i]
                        delta_theta = np.arctan2(np.sin(delta_theta), np.cos(delta_theta))
                        
                        # Gaussian repulsion
                        gaussian_repulsion = k_rep * (-delta_theta / sigma**2) * np.exp(-(delta_theta**2) / (2*sigma**2))
                        angular_forces[i] += gaussian_repulsion

        return angular_forces
    
    def update_viewpoints(self, viewpoints, angular_positions, learning_rate=0.01,
                      m=None, v=None, t=0, beta1=0.9, beta2=0.999, eps=1e-8, 
                      k_attr=0.4, k_rep=1):
        """
        Update viewpoint positions using angular coordinates with Adam optimizer.
        """
        # Calculate angular forces directly from projected object points
        angular_forces = self.compute_angular_forces(angular_positions, 
                                               alpha=k_attr, k_rep=k_rep)
        
        # Apply damping to reduce oscillations
        if t > 50:  # After initial exploration phase
            angular_forces *= 0.9  # Damping factor
        
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
field = AngularViewPlanningField(grid_size, num_particles, fov_angle, fov_radius)
object_points = field.set_object_points(object_vertices, num_points_per_edge=10)
viewpoints, angular_positions = field.initialize_viewpoints(num_particles, circle_radius, circle_center)

# Calculate initial visibility and potential
coverage = field.update_visibility(viewpoints)
field.compute_potential(viewpoints)

# Setup figure for animation
fig = plt.figure(figsize=(18, 6), dpi=100)
gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 1])

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

# Energy plot
ax_energy = fig.add_subplot(gs[0, 3])
ax_energy.set_xlim(0, frames)
ax_energy.set_ylim(0, 1)
ax_energy.set_xlabel("Time Step", fontsize=10)
ax_energy.set_ylabel("Potential Energy", fontsize=10)
ax_energy.set_title("Potential Energy Over Time", fontsize=10)

energy_line, = ax_energy.plot([], [], color='red')

# Adjust spacing
plt.subplots_adjust(wspace=0.3)

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
object_scatter = ax_main.scatter(object_points[:, 0], object_points[:, 1], 
                               c=['red' if v == 0 else 'green' for v in field.obj_visibility], 
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

# Potential energy data
potential_energy_data = []

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
    
    # Second, compute potential based on visibility
    field.compute_potential(viewpoints)

    # Compute total potential energy (sum of pairwise distances)
    positions = viewpoints[:, :2]
    diffs = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(distances, 0)
    energy = np.sum(distances) / 2.0  # Divide by 2 to correct double-counting
    potential_energy_data.append(energy)
    
    # Update max potential for color scaling
    current_max = np.max(field.obj_potentials)
    if current_max > 0:
        max_potential_seen = max(max_potential_seen, current_max)
    
    # Update viewpoints with fixed learning rate
    viewpoints, angular_positions, m_angular, v_angular, t_angular = field.update_viewpoints(
        viewpoints, angular_positions, learning_rate=0.05, 
        m=m_angular, v=v_angular, t=t_angular,
        k_attr=1.0, k_rep=0.2
    )
    
    # Update visualization
    
    # 1. Update object points coloring by visibility
    visibility_colors = plt.cm.RdYlGn(field.obj_visibility)  # Red to yellow to green
    object_scatter.set_color(visibility_colors)
    
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
    
    # 5. Update energy plot
    energy_line.set_data(range(len(potential_energy_data)), potential_energy_data)
    ax_energy.set_ylim(0, max(1.0, max(potential_energy_data)*1.1))  # Dynamic y-scaling
    if len(potential_energy_data) > ax_energy.get_xlim()[1]:
        ax_energy.set_xlim(0, len(potential_energy_data) + 1)

    # 6. Update the projected points on the manifold in the main view
    projected_x = circle_center[0] + circle_radius * np.cos(field.obj_manifold_projections)
    projected_y = circle_center[1] + circle_radius * np.sin(field.obj_manifold_projections)
    
    manifold_projected_scatter.set_offsets(np.column_stack([projected_x, projected_y]))
    manifold_projected_scatter.set_array(field.obj_potentials)
    manifold_projected_scatter.set_clim(vmin=0, vmax=max(max_potential_seen, 0.01))
    
    # 7. Update manifold visualization with object projections in the graph
    manifold_object_scatter.set_offsets(np.column_stack([
        field.obj_manifold_projections, 
        field.obj_potentials
    ]))
    
    # Update colors to match potentials
    manifold_object_scatter.set_array(field.obj_potentials)
    
    # Update color normalization
    manifold_object_scatter.set_clim(vmin=0, vmax=max(max_potential_seen, 0.01))
    ax_manifold.set_ylim(0, max(max_potential_seen*1.1, 0.01))  # Dynamic y-axis scaling
    
    # 8. Update viewpoint positions on manifold
    manifold_viewpoint_scatter.set_offsets(np.column_stack([
        angular_positions, 
        np.zeros_like(angular_positions)
    ]))
    
    # 9. Update stats text
    elapsed = time.time() - start_time
    stats_text.set_text(f"Frame: {frame}\nCoverage: {coverage:.2f}\nTime: {elapsed:.3f}s")
    
    return [object_scatter, viewpoint_scatter, manifold_projected_scatter, manifold_object_scatter, 
            manifold_viewpoint_scatter, coverage_line, energy_line, stats_text]

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=False)

plt.tight_layout()
plt.show()
# ani.save('view_planning_direct_projection.mp4', writer='ffmpeg', fps=120, dpi=100)