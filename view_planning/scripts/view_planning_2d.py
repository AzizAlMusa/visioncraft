import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Wedge, Circle
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

# Configuration parameters
grid_size = 100  # Grid resolution for potential field calculation
frames = 200     # Number of animation frames
num_particles = 8  # Initial number of viewpoints

# Circle parameters
circle_radius = 30  # Radius of the bounding circle for viewpoints
circle_center = np.array([grid_size/2, grid_size/2])  # Center of the circle

# Field of view parameters
fov_angle = 60  # Field of view angle in degrees
fov_radius = 20  # Maximum distance a viewpoint can see

# Object parameters - we'll use a simple polygon for the observable object
# Object parameters - an irregular polygon without deep recesses
# Object parameters - an irregular polygon with asymmetric vertex distribution
object_vertices = np.array([
    [40, 35],   # Starting at bottom left
    [55, 32],   # Bottom right
    [62, 38],
    [65, 45],
    [67, 52],   # Multiple points concentrated on the right side
    [66, 58],
    [64, 64],
    [60, 67],
    [54, 69],   # Top right
    [45, 70],   # Top middle
    [35, 65],   # Top left
    [30, 55],
    [28, 45],   # Left middle
])
# Create a mesh grid for the potential field calculation
x = np.arange(grid_size)
y = np.arange(grid_size)
X, Y = np.meshgrid(x, y)
field_points = np.stack([X.ravel(), Y.ravel()], axis=-1)

# Small value to prevent division by zero
epsilon = 1e-6

class ViewPlanningField:
    def __init__(self, grid_size, num_particles, fov_angle, fov_radius):
        self.grid_size = grid_size
        self.fov_angle = fov_angle  # in degrees
        self.fov_radius = fov_radius
        
        # Visibility array: 1 if covered, 0 otherwise
        self.visibility = np.zeros((grid_size, grid_size), dtype=np.float64)
        
        # Potential array
        self.potential = np.zeros((grid_size, grid_size))
        
        # Grid points in flattened form (grid_size*grid_size, 2)
        self.field_points = field_points
        
        # Forces for visualization (grid_size, grid_size, num_particles, 2)
        self.attractive_forces = np.zeros(
            (grid_size, grid_size, num_particles, 2), dtype=np.float64
        )
        
        # (num_particles, num_particles, 2)
        self.repulsive_forces = np.zeros((num_particles, num_particles, 2), dtype=np.float64)
        
        # Object with vertices
        self.object_points = None
        
    def set_object_points(self, vertices, num_points_per_edge=10):
        """
        Convert object vertices to a set of points along the perimeter.
        For a polygon, we sample points along each edge.
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
        return self.object_points
    
    def initialize_viewpoints_on_circle(self, num_viewpoints, radius, center):
        """
        Initialize viewpoints evenly distributed on a circle.
        All viewpoints point toward the center.
        """
        angles = np.linspace(0, 0.1, num_viewpoints, endpoint=False)
        
        # Position on circle
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        
        # Direction vectors (pointing to center)
        dx = center[0] - x
        dy = center[1] - y
        
        # Normalize direction vectors
        norm = np.sqrt(dx**2 + dy**2) + epsilon
        dx /= norm
        dy /= norm
        
        # Combine position and direction
        viewpoints = np.column_stack([x, y, dx, dy])
        return viewpoints
    
    def update_visibility(self, viewpoints, beta=20.0):
        """
        Update visibility based on viewpoint FOVs.
        Each viewpoint has position (x,y) and direction (dx,dy).
        """
        self.visibility.fill(0)
        
        # Extract positions and directions
        positions = viewpoints[:, :2]  # (x, y)
        directions = viewpoints[:, 2:]  # (dx, dy)
        
        # For each grid point
        diff = self.field_points[:, None, :] - positions[None, :, :]
        distances = np.linalg.norm(diff, axis=-1) + epsilon
        
        # Check if within FOV radius
        in_radius = distances <= self.fov_radius
        
        # Calculate angle between viewpoint direction and point
        # Normalize the difference vectors
        diff_norm = diff / distances[:, :, None]
        
        # Dot product with viewpoint direction gives cosine of angle
        cos_angles = np.sum(diff_norm * directions[None, :, :], axis=2)
        
        # Convert FOV angle to radians and calculate its cosine
        fov_rad = np.radians(self.fov_angle / 2)  # Half-angle
        cos_fov = np.cos(fov_rad)
        
        # Point is in FOV if cos(angle) > cos(fov/2)
        in_fov = cos_angles > cos_fov
        
        # Soft visibility function using sigmoid
        visibility_factor = 1.0 / (1.0 + np.exp(beta * (distances / self.fov_radius - 1.0)))
        
        # Apply FOV constraint
        visibility_factor = visibility_factor * in_fov
        
        # Aggregate visibility across all viewpoints
        self.visibility.ravel()[:] = np.clip(np.sum(visibility_factor, axis=1), 0.0, 1.0)
        
        # Return coverage as the fraction of object points that are visible
        if self.object_points is not None:
            # Get object point indices in the grid
            obj_indices = np.round(self.object_points).astype(int)
            obj_indices = np.clip(obj_indices, 0, self.grid_size - 1)
            
            # Get visibility values at object points
            obj_visibility = self.visibility[obj_indices[:, 1], obj_indices[:, 0]]
            coverage = np.mean(obj_visibility > 0.5)
            return coverage
        
        return 0.0
    
    def compute_potential(self, viewpoints, alpha=1.0):
        """
        Compute potential field based on log distance, weighted by coverage deficit.
        Only object points contribute to the potential.
        """
        # Reset potential
        self.potential.fill(0)
        
        if self.object_points is None:
            return
        
        # Project object points to grid
        obj_indices = np.round(self.object_points).astype(int)
        obj_indices = np.clip(obj_indices, 0, self.grid_size - 1)
        
        # Get visibility values at object points
        obj_visibility = self.visibility[obj_indices[:, 1], obj_indices[:, 0]]
        
        # Coverage deficit for each object point
        coverage_deficit = 1.0 - obj_visibility
        
        # For each grid point, compute log potential from uncovered object points
        positions = viewpoints[:, :2]
        
        for i, (x, y) in enumerate(self.field_points):
            point = np.array([x, y])
            
            # Skip points far from the bounding circle
            dist_to_center = np.linalg.norm(point - circle_center)
            if abs(dist_to_center - circle_radius) > 2:
                continue
                
            # Calculate potential from uncovered object points
            diffs = self.object_points - point[None, :]
            distances = np.linalg.norm(diffs, axis=1) + epsilon
            
            # Weight by coverage deficit
            log_pot = np.sum(coverage_deficit * np.log(distances))
            
            # Store in potential grid
            ix, iy = int(x), int(y)
            if 0 <= ix < self.grid_size and 0 <= iy < self.grid_size:
                self.potential[iy, ix] = alpha * log_pot
    
    def compute_attractive_force(self, viewpoints, alpha=1.0):
        """
        Compute attractive forces based on gradient of log potential.
        Forces act only on the positions, not on directions.
        """
        positions = viewpoints[:, :2]
        num_viewpoints = positions.shape[0]
        
        # Reset forces
        attractive_forces = np.zeros((num_viewpoints, 2))
        
        if self.object_points is None:
            return attractive_forces
            
        # Get object points with coverage deficit
        obj_indices = np.round(self.object_points).astype(int)
        obj_indices = np.clip(obj_indices, 0, self.grid_size - 1)
        obj_visibility = self.visibility[obj_indices[:, 1], obj_indices[:, 0]]
        coverage_deficit = 1.0 - obj_visibility
        
        # Only consider object points with deficit
        uncovered_idx = coverage_deficit > 0.1
        uncovered_points = self.object_points[uncovered_idx]
        uncovered_deficit = coverage_deficit[uncovered_idx]
        
        if len(uncovered_points) == 0:
            return attractive_forces
            
        # For each viewpoint
        for i, pos in enumerate(positions):
            # Calculate forces from uncovered object points
            diffs = uncovered_points - pos[None, :]
            distances = np.linalg.norm(diffs, axis=1) + epsilon
            
            # Normalize to get direction
            directions = diffs / distances[:, None]
            
            # Force is proportional to coverage deficit and inversely to distance
            force_magnitudes = alpha * uncovered_deficit / distances
            
            # Calculate total force on this viewpoint
            force_vectors = force_magnitudes[:, None] * directions
            attractive_forces[i] = np.sum(force_vectors, axis=0)
            
        return attractive_forces
    
    def compute_repulsive_force(self, viewpoints, sigma=10, amplitude=100):
        """
        Compute Gaussian repulsion between viewpoints.
        Similar to your playground implementation.
        """
        positions = viewpoints[:, :2]
        num_viewpoints = positions.shape[0]
        
        # Reset forces
        repulsive_forces = np.zeros((num_viewpoints, 2))
        
        if num_viewpoints <= 1:
            return repulsive_forces
            
        # Pairwise differences
        pairwise_diff = positions[:, None, :] - positions[None, :, :]
        pairwise_distances = np.linalg.norm(pairwise_diff, axis=2) + epsilon
        
        # Gaussian repulsion
        repulsion = amplitude * np.exp(-pairwise_distances**2 / (2 * sigma**2))
        
        # Force direction is along the difference vector
        rep_forces = np.zeros((num_viewpoints, num_viewpoints, 2))
        for i in range(num_viewpoints):
            for j in range(num_viewpoints):
                if i != j:
                    direction = pairwise_diff[i, j] / pairwise_distances[i, j]
                    rep_forces[i, j] = repulsion[i, j] * direction
        
        # Sum repulsion from all other viewpoints
        repulsive_forces = np.sum(rep_forces, axis=1)
        
        return repulsive_forces
        
    def compute_total_force(self, viewpoints, k_attr=0.4, k_rep=0.0, alpha=1.0, sigma=10, amplitude=100):
        """
        Combine attractive and repulsive forces.
        """
        attractive = self.compute_attractive_force(viewpoints, alpha)
        repulsive = self.compute_repulsive_force(viewpoints, sigma, amplitude)
        
        return k_attr * attractive + k_rep * repulsive
        
    def constrain_to_circle(self, positions, radius, center):
        """
        Constrain positions to lie on the circle.
        """
        # Vector from center to position
        vectors = positions - center[None, :]
        
        # Normalize to radius
        distances = np.linalg.norm(vectors, axis=1) + epsilon
        normalized = vectors / distances[:, None] * radius
        
        # New positions
        new_positions = center[None, :] + normalized
        
        return new_positions
        
    def update_directions(self, viewpoints, center):
        """
        Update direction vectors to point toward the center.
        """
        positions = viewpoints[:, :2]
        
        # Direction to center
        vectors = center[None, :] - positions
        
        # Normalize
        distances = np.linalg.norm(vectors, axis=1) + epsilon
        directions = vectors / distances[:, None]
        
        # Update viewpoints
        viewpoints[:, 2:] = directions
        
        return viewpoints
        
    def update_viewpoints(self, viewpoints, learning_rate=1.0, m=None, v=None, t=0, 
                         beta1=0.9, beta2=0.999, eps=1e-8, 
                         k_attr=0.4, k_rep=0.2):
        """
        Update viewpoint positions using Adam optimizer.
        Constrain positions to the bounding circle and update directions.
        """
        # Calculate forces
        forces = self.compute_total_force(viewpoints, k_attr, k_rep)
        
        # Apply forces only to positions, not directions
        grad = np.zeros_like(viewpoints)
        grad[:, :2] = forces
        
        # Adam optimizer update
        if m is None:
            m = np.zeros_like(viewpoints)
        if v is None:
            v = np.zeros_like(viewpoints)
            
        t += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        # Update positions
        viewpoints += learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        
        # Constrain positions to circle
        viewpoints[:, :2] = self.constrain_to_circle(viewpoints[:, :2], circle_radius, circle_center)
        
        # Update directions to point to center
        viewpoints = self.update_directions(viewpoints, circle_center)
        
        return viewpoints, m, v, t


# Initialize the field
field = ViewPlanningField(grid_size, num_particles, fov_angle, fov_radius)

# Create object points from vertices
object_points = field.set_object_points(object_vertices, num_points_per_edge=10)

# Initialize viewpoints on circle
viewpoints = field.initialize_viewpoints_on_circle(num_particles, circle_radius, circle_center)

# Setup figure for animation
fig = plt.figure(figsize=(16, 8), dpi=100)
gs = GridSpec(1, 2, width_ratios=[1, 1])

# Main visualization area
ax_main = fig.add_subplot(gs[0, 0])
ax_main.set_xlim(0, grid_size)
ax_main.set_ylim(0, grid_size)
ax_main.set_aspect("equal")
ax_main.set_title("View Planning Simulation", fontsize=12)

# Line plot for coverage
ax_line = fig.add_subplot(gs[0, 1])
ax_line.set_xlim(0, frames)
ax_line.set_ylim(0, 1)
ax_line.set_xlabel("Time Step", fontsize=10)
ax_line.set_ylabel("Coverage", fontsize=10)
ax_line.set_title("Coverage Over Time", fontsize=10)

# Adjust spacing
plt.subplots_adjust(wspace=0.3)

# Initial field calculation
coverage = field.update_visibility(viewpoints)
field.compute_potential(viewpoints)

# Plot elements
# Contour for potential field
contour = ax_main.contourf(X, Y, field.potential, levels=50, cmap="viridis", alpha=0.7, origin="lower")

# Bounding circle
circle = plt.Circle(circle_center, circle_radius, fill=False, color='black', linestyle='--')
ax_main.add_patch(circle)

# Object as polygon
polygon = plt.Polygon(object_vertices, fill=True, alpha=0.3, color='gray')
ax_main.add_patch(polygon)

# Object points
object_scatter = ax_main.scatter(object_points[:, 0], object_points[:, 1], 
                               c='red', s=10, alpha=0.5, zorder=2)

# Viewpoints as scatter
viewpoint_scatter = ax_main.scatter(viewpoints[:, 0], viewpoints[:, 1], 
                                 c='blue', s=50, zorder=3)

# FOV visualization as wedges
fov_patches = []
for i in range(viewpoints.shape[0]):
    pos = viewpoints[i, :2]
    direction = viewpoints[i, 2:]
    
    # Calculate angle in degrees
    angle = np.degrees(np.arctan2(direction[1], direction[0]))
    
    # Create wedge
    wedge = Wedge(pos, fov_radius, angle - fov_angle/2, angle + fov_angle/2, 
                 alpha=0.2, color='blue', zorder=1)
    ax_main.add_patch(wedge)
    fov_patches.append(wedge)

# Coverage line
coverage_line, = ax_line.plot([], [], color='blue')
coverage_data = [coverage]

# Colorbar for potential
cbar = plt.colorbar(contour, ax=ax_main)
cbar.set_label('Potential Field')

# Text for stats
stats_text = ax_main.text(0.02, 0.98, "", transform=ax_main.transAxes, 
                        ha='left', va='top', fontsize=10)

# Initialize optimizer variables
m = np.zeros_like(viewpoints)
v = np.zeros_like(viewpoints)
t = 0

def update(frame):
    global viewpoints, m, v, t, contour, fov_patches
    
    start_time = time.time()
    
    # Update visibility and calculate coverage
    coverage = field.update_visibility(viewpoints)
    coverage_data.append(coverage)
    
    # Compute potential field for visualization
    field.compute_potential(viewpoints)
    
    # Update viewpoints with Adam optimizer
    viewpoints, m, v, t = field.update_viewpoints(
        viewpoints, learning_rate=2.0, m=m, v=v, t=t,
        k_attr=0.4, k_rep=0.2
    )
    
    # Update visualization
    # 1. Update contour
    for c in contour.collections:
        c.remove()
    contour = ax_main.contourf(X, Y, field.potential, levels=50, 
                             cmap="viridis", alpha=0.7, origin="lower")
    
    # 2. Update viewpoint positions
    viewpoint_scatter.set_offsets(viewpoints[:, :2])
    
    # 3. Update FOV wedges
    for i, wedge in enumerate(fov_patches):
        wedge.remove()
    
    fov_patches = []
    for i in range(viewpoints.shape[0]):
        pos = viewpoints[i, :2]
        direction = viewpoints[i, 2:]
        
        # Calculate angle in degrees
        angle = np.degrees(np.arctan2(direction[1], direction[0]))
        
        # Create wedge
        wedge = Wedge(pos, fov_radius, angle - fov_angle/2, angle + fov_angle/2, 
                     alpha=0.2, color='blue', zorder=1)
        ax_main.add_patch(wedge)
        fov_patches.append(wedge)
    
    # 4. Update coverage line
    coverage_line.set_data(range(len(coverage_data)), coverage_data)
    
    # 5. Update stats text
    elapsed = time.time() - start_time
    stats_text.set_text(f"Frame: {frame}\nCoverage: {coverage:.2f}\nTime: {elapsed:.3f}s")
    
    # Adjust axis if needed
    ax_line.set_xlim(0, max(len(coverage_data), frames))
    
    return [viewpoint_scatter, coverage_line, stats_text]

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=False)

plt.tight_layout()
plt.show()

# Uncomment to save animation
# ani.save('view_planning.mp4', writer='ffmpeg', fps=30, dpi=200)