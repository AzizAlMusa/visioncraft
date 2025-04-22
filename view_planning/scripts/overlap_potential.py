import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

# Number of particles and grid size
grid_size = 100
frames = 600

# When to add/remove viewpoints (frame number)
REMOVE_FRAMES = []       # Frames to remove viewpoints
ADD_FRAMES = []          # Frames to add viewpoints 50, 100, 150, 200, 250, 300, 350, 400, 450, 500
NEW_PARTICLES = 1        # Number of new viewpoints to add/remove

# Initial positions of particles
num_particles = 7
particles = 50 + np.random.rand(num_particles, 2)

# Create a mesh grid for the potential field
x = np.arange(grid_size)
y = np.arange(grid_size)
X, Y = np.meshgrid(x, y)
field_points = np.stack([X.ravel(), Y.ravel()], axis=-1)

epsilon = 1e-6

class Field:
    def __init__(self, grid_size, num_particles, fov_radius, use_wrapping=True):
        self.grid_size = grid_size
        self.fov_radius = fov_radius
        self.use_wrapping = use_wrapping

        # Visibility array: values represent coverage quality
        self.visibility = np.zeros((grid_size, grid_size), dtype=np.float64)
        
        # Overlap count array: tracks how many particles cover each point
        self.overlap_count = np.zeros((grid_size, grid_size), dtype=np.int32)
        
        # Optimal overlap array: tracks areas with ideal coverage (covered exactly twice)
        self.optimal_overlap = np.zeros((grid_size, grid_size), dtype=np.float64)

        # Potential array
        self.potential = np.zeros((grid_size, grid_size))

        # Grid points in flattened form (grid_size*grid_size, 2)
        self.field_points = field_points

        # Forces for visualization
        # (grid_size, grid_size, num_particles, 2)
        self.attractive_forces = np.zeros(
            (grid_size, grid_size, num_particles, 2), dtype=np.float64
        )
        # (num_particles, num_particles, 2)
        self.repulsive_forces = np.zeros((num_particles, num_particles, 2), dtype=np.float64)
        
        # Calculate theoretical maximum coverage
        self.theoretical_max_coverage = min(1.0, num_particles * np.pi * (fov_radius**2) / (grid_size**2))

    def compute_coverage_monte_carlo(self, particles, num_samples=10000, threshold=0.0):
        """
        Computes coverage using Monte Carlo sampling with physically accurate metrics.
        A point is considered covered if it's within radius of at least one particle.
        """
        # Sample random points in the space
        random_points = np.random.rand(num_samples, 2) * self.grid_size

        # Compute distances to particles
        diff = random_points[:, None, :] - particles[None, :, :]
        wrapped_diff = self.wrap_distance(diff)
        distances = np.linalg.norm(wrapped_diff, axis=-1)

        # Binary coverage: a point is covered if within radius of any particle
        is_covered_by_particle = distances <= self.fov_radius
        covered_points = np.any(is_covered_by_particle, axis=1)
        coverage_fraction = np.mean(covered_points)
        
        # Calculate overlap counts - how many particles cover each point
        overlap_counts = np.sum(is_covered_by_particle, axis=1)
        
        # Calculate overlap statistics on covered points only
        if np.sum(covered_points) > 0:
            # For covered points only
            covered_overlap = overlap_counts[covered_points]
            
            # Exact counts (no approximations)
            single_coverage = np.mean(covered_overlap == 1)  # Fraction covered by exactly 1
            double_coverage = np.mean(covered_overlap == 2)  # Fraction covered by exactly 2
            excess_coverage = np.mean(covered_overlap > 2)   # Fraction covered by more than 2
            
            # Calculate optimal coverage ratio (% of covered area with 1-2 particles)
            optimal_area = np.sum((covered_overlap == 1) | (covered_overlap == 2))
            optimal_ratio = optimal_area / len(covered_overlap)
                
            # Calculate redundancy ratio (% of "particle coverage" that is redundant)
            total_coverage_units = np.sum(overlap_counts)
            necessary_coverage_units = np.sum(covered_points)
            redundant_units = total_coverage_units - necessary_coverage_units
            
            if total_coverage_units > 0:
                redundancy_ratio = redundant_units / total_coverage_units
            else:
                redundancy_ratio = 0.0
        else:
            single_coverage = double_coverage = excess_coverage = 0.0
            optimal_ratio = redundancy_ratio = 0.0

        # Recalculate theoretical max for current number of particles
        num_particles = particles.shape[0]
        theoretical_max = min(1.0, num_particles * np.pi * (self.fov_radius**2) / (self.grid_size**2))
        self.theoretical_max_coverage = theoretical_max
        
        # Verify coverage doesn't exceed theoretical maximum
        if coverage_fraction > theoretical_max + 0.01:  # Allow small float error
            print(f"WARNING: Calculated coverage {coverage_fraction:.4f} exceeds theoretical maximum {theoretical_max:.4f}")
            coverage_fraction = theoretical_max

        return {
            'coverage': coverage_fraction,
            'single_coverage': single_coverage,
            'double_coverage': double_coverage,
            'excess_coverage': excess_coverage,
            'optimal_ratio': optimal_ratio,
            'redundancy_ratio': redundancy_ratio,
            'theoretical_max': theoretical_max
        }
    
    def wrap_distance(self, diff):
        """
        Compute toroidal wrapping so distance remains within [-grid_size/2, grid_size/2].
        If wrapping is disabled, diff is unchanged.
        """
        if not self.use_wrapping:
            return diff
        # If |diff| > grid_size/2, wrap around the other side
        return np.where(
            np.abs(diff) > self.grid_size / 2,
            -np.sign(diff) * (self.grid_size - np.abs(diff)),
            diff
        )

    def update_visibility(self, particles):
        """
        Calculate visibility values using a properly implemented step function approximation.
        This creates a continuous, differentiable approximation of step function that
        assigns different qualities to different coverage levels.
        """
        # Calculate effective coverage
        diff = self.field_points[:, None, :] - particles[None, :, :]
        wrapped_diff = self.wrap_distance(diff)
        distances = np.linalg.norm(wrapped_diff, axis=-1)
        
        # Sigmoid function for smooth particle contributions
        decay_rate = 0.3
        particle_contributions = 1.0 / (1.0 + np.exp((distances - self.fov_radius) * decay_rate))
        effective_coverage = np.sum(particle_contributions, axis=1)
        
        # Update binary overlap for metrics
        self.update_binary_overlap(particles)
        
        # Define coverage quality values - these can be adjusted
        # Format: [(coverage_level, quality_value), ...]
        coverage_qualities = [
            (0.0, 0.2),  # No coverage: value = 0.3
            (1.0, 0.0),  # Single coverage: value = 0.8
            (2.0, 1.0),  # Double coverage: value = 1.0
            (3.0, 1.0),  # Triple coverage: value = 0.5
            (4.0, 1.0)   # Four+ coverage: value = 0.2
        ]
        
        # Sharpness of transitions
        sharpness = 12.0
        
        # Calculate the quality using a smooth approximation of a step function
        quality_values = np.zeros_like(effective_coverage)
        
        # Apply each step
        for i in range(len(coverage_qualities)):
            level, value = coverage_qualities[i]
            
            if i == len(coverage_qualities) - 1:
                # Last level - apply to all points above this level
                weight = 1.0 / (1.0 + np.exp(-sharpness * (effective_coverage - level)))
                quality_values += value * weight
            else:
                # Calculate window between this level and the next
                next_level = coverage_qualities[i+1][0]
                
                # Calculate a "window" function that's ~1 within this range and ~0 elsewhere
                # Rising edge at current level
                rising_edge = 1.0 / (1.0 + np.exp(-sharpness * (effective_coverage - level)))
                # Falling edge at next level
                falling_edge = 1.0 / (1.0 + np.exp(-sharpness * (effective_coverage - next_level)))
                
                window = rising_edge * (1.0 - falling_edge)
                quality_values += value * window
        
        # Ensure quality values stay within [0, 1]
        quality_values = np.clip(quality_values, 0.0, 1.0)
        
        # Reshape to grid dimensions
        self.visibility = quality_values.reshape(self.grid_size, self.grid_size)
    
    def update_binary_overlap(self, particles):
        """
        Updates the binary overlap count for physical accuracy.
        This is used for visualization and metrics.
        """
        diff = self.field_points[:, None, :] - particles[None, :, :]
        wrapped_diff = self.wrap_distance(diff)
        distances = np.linalg.norm(wrapped_diff, axis=-1)
        
        # Binary coverage: 1 if within radius, 0 otherwise
        is_covered = distances <= self.fov_radius
        overlap_count = np.sum(is_covered, axis=1)
        
        # Reshape to grid dimensions
        self.overlap_count = overlap_count.reshape(self.grid_size, self.grid_size)
        
        # Calculate optimal overlap (areas covered exactly twice)
        self.optimal_overlap = (overlap_count == 2).astype(float).reshape(self.grid_size, self.grid_size)

    # def compute_potential(self, particles, alpha=1.0):
    #     """
    #     A modified potential based on a continuous, differentiable coverage need function.
    #     The potential is:
    #     1. Highest for uncovered areas (to encourage coverage)
    #     2. Low for optimally covered areas (visibility near 1.0)
    #     3. Increases again for over-covered areas (to discourage redundancy)
        
    #     Uses (1-visibility) as a smooth, continuous factor to guide particles.
    #     """
    #     diff = self.field_points[:, None, :] - particles[None, :, :]
    #     wrapped_diff = self.wrap_distance(diff)
    #     distances = np.linalg.norm(wrapped_diff, axis=-1) + epsilon

    #     # Sum of log(distance) across all particles
    #     log_sum = np.log(distances).sum(axis=1)  # (grid_size*grid_size,)

    #     # Coverage need is (1 - visibility) since our visibility function 
    #     # already encodes the desired behavior
    #     coverage_need = 1.0 - self.visibility.ravel()

    #     # Final potential
    #     pot_values = alpha * coverage_need * log_sum
    #     self.potential = pot_values.reshape(self.grid_size, self.grid_size)

    # def compute_attractive_force(self, particles, alpha=1.0):
    #     """
    #     Negative gradient of the modified potential field.
    #     Uses the continuous (1-visibility) factor as the coverage need,
    #     resulting in a smooth, differentiable force field.
    #     """
    #     diff = self.field_points[:, None, :] - particles[None, :, :]
    #     wrapped_diff = self.wrap_distance(diff)
    #     distances = np.linalg.norm(wrapped_diff, axis=-1) + epsilon
    #     directions = wrapped_diff / distances[..., None]  # shape: (Npoints, Nparticles, 2)

    #     # Coverage need is (1 - visibility) - smooth and differentiable
    #     coverage_need = (1.0 - self.visibility.ravel())[:, None]  # shape: (Npoints, 1)

    #     # Combine factors: alpha * coverage_need / distance
    #     force_magnitude = alpha * coverage_need / distances
    #     force_magnitude[distances < epsilon] = 0.0

    #     # Force vector from each grid point to each particle
    #     attractive_forces = force_magnitude[..., None] * directions

    #     # Reshape to (grid_size, grid_size, num_particles, 2) for visualization
    #     self.attractive_forces = attractive_forces.reshape(
    #         self.grid_size, self.grid_size, particles.shape[0], 2
    #     )

    #     # Sum over all grid points => shape (num_particles, 2)
    #     total_attractive = self.attractive_forces.sum(axis=(0, 1))
    #     return total_attractive
    def compute_potential(self, particles, alpha=1.0, sigma=10.0):
        """
        Information-theoretic potential:
        φ(r) = -0.5 * log(1 + exp(-r^2 / σ^2) / σ^2)

        Encourages spacing and attraction to uncovered regions without collapse.
        """
        diff = self.field_points[:, None, :] - particles[None, :, :]  # (Npoints, Nparticles, 2)
        wrapped_diff = self.wrap_distance(diff)
        distances_sq = np.sum(wrapped_diff**2, axis=-1)  # shape: (Npoints, Nparticles)

        # Information-theoretic term
        info = -0.5 * np.log1p(np.exp(-distances_sq / sigma**2) / sigma**2)  # (Npoints, Nparticles)
        info_sum = info.sum(axis=1)  # sum over particles

        covered_factor = 1.0 - self.visibility.ravel()
        pot_values = alpha * covered_factor * info_sum

        self.potential = pot_values.reshape(self.grid_size, self.grid_size)


    def compute_attractive_force(self, particles, alpha=1.0, sigma=10.0):
        """
        Gradient of the information-theoretic potential:
        F ∝ [exp(-r^2 / σ^2) / (σ^4 * (1 + exp(-r^2 / σ^2)/σ^2))] * (p - x)

        Produces stable attraction toward uncovered regions, no collapse.
        """
        diff = self.field_points[:, None, :] - particles[None, :, :]  # (Npoints, Nparticles, 2)
        wrapped_diff = self.wrap_distance(diff)
        distances_sq = np.sum(wrapped_diff**2, axis=-1)  # (Npoints, Nparticles)

        exp_term = np.exp(-distances_sq / sigma**2)
        denom = sigma**4 * (1 + exp_term / sigma**2)
        factor = exp_term / denom  # (Npoints, Nparticles)

        covered_factor = (1.0 - self.visibility.ravel())[:, None]  # (Npoints, 1)
        force_mags = alpha * covered_factor * factor  # (Npoints, Nparticles)

        attractive_forces = force_mags[..., None] * wrapped_diff  # (Npoints, Nparticles, 2)

        self.attractive_forces = attractive_forces.reshape(
            self.grid_size, self.grid_size, particles.shape[0], 2
        )

        total_attractive = self.attractive_forces.sum(axis=(0, 1))  # (Nparticles, 2)
        return total_attractive
    
    def compute_repelling_force(self, particles, sigma=10, amplitude=100):
        """
        Particle-particle repulsion. Gaussian-based, so it decays softly
        with distance, allowing some overlap but discouraging collisions.
        """
        pairwise_diff = particles[:, None, :] - particles[None, :, :]
        wrapped_pairwise_diff = self.wrap_distance(pairwise_diff)
        pairwise_distances = np.linalg.norm(wrapped_pairwise_diff, axis=-1) + epsilon

        # Gaussian-based repulsion
        repelling_forces = -(
            amplitude
            * wrapped_pairwise_diff
            * (-pairwise_distances[..., None] / sigma**2)
            * np.exp(-pairwise_distances**2 / (2 * sigma**2))[..., None]
        )

        self.repulsive_forces = repelling_forces
        # Sum repulsion from all other particles
        total_repelling = repelling_forces.sum(axis=1)  # (num_particles, 2)
        return total_repelling

    def compute_force(self, particles, sigma=10, amplitude=100,
                      k_attr=0.4, k_rep=0.05, alpha=1.0):
        """
        Combine attraction and repulsion:
          F_total = k_attr * F_attr + k_rep * F_rep
        
        Added small repulsion by default to avoid particles converging too closely.
        """
        total_attractive = self.compute_attractive_force(particles, alpha)
        total_repelling = self.compute_repelling_force(particles, sigma, amplitude)
        return k_attr * total_attractive + k_rep * total_repelling


# Initialize the field
field = Field(grid_size, num_particles, fov_radius=20)

# === FIGURE SETUP ===
fig = plt.figure(figsize=(15, 8), dpi=150)
gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[3, 1])

# Main plot in top left
ax_main = fig.add_subplot(gs[0, 0])
# Coverage plot in top right
ax_line = fig.add_subplot(gs[0, 1])
# Overlap metrics plot at bottom
ax_overlap = fig.add_subplot(gs[1, :])

plt.subplots_adjust(wspace=0.3, hspace=0.3)

# === LEFT PLOT: Field Visualization ===
ax_main.set_xlim(0, grid_size - 1)
ax_main.set_ylim(0, grid_size - 1)
ax_main.set_title("Field Potential & Coverage", fontsize=12)
ax_main.set_aspect("equal")

# Contour plot with automatic colorbar
contour = ax_main.contourf(
    X, Y, field.potential, 
    levels=100, cmap="viridis", alpha=0.9, origin="lower"
)

# Particle styling
scatter = ax_main.scatter(
    particles[:, 0], particles[:, 1],
    c="deepskyblue", s=50, zorder=5
)

# We'll create circles in the update function instead of here
# This will ensure they're properly managed between frames
circles = []

# === TOP RIGHT PLOT: Coverage Growth Over Time ===
ax_line.set_xlim(0, frames)
ax_line.set_ylim(0, 1)
ax_line.set_xlabel("Time Step", fontsize=10)
ax_line.set_ylabel("Coverage", fontsize=10)
ax_line.set_title("Coverage Over Time", fontsize=10)

# Coverage line
coverage_line, = ax_line.plot([], [], color="#3d03fc", label="Coverage")
max_coverage_line, = ax_line.plot([], [], 'r--', label="Max Theoretical")

# === BOTTOM PLOT: Overlap Metrics ===
ax_overlap.set_xlim(0, frames)
ax_overlap.set_ylim(0, 1)
ax_overlap.set_xlabel("Time Step", fontsize=10)
ax_overlap.set_ylabel("Ratio", fontsize=10)
ax_overlap.set_title("Overlap Metrics", fontsize=10)

# Lines for each metric
single_line, = ax_overlap.plot([], [], 'g-', label="Single")
double_line, = ax_overlap.plot([], [], 'b-', label="Double (Optimal)")
excess_line, = ax_overlap.plot([], [], 'r-', label="Excess (>2)")
redundancy_line, = ax_overlap.plot([], [], 'k--', label="Redundancy Ratio")

ax_overlap.legend(loc='upper right', ncol=4, fontsize=8)
ax_line.legend(loc='upper right', fontsize=8)

# Initial computations
field.update_visibility(particles)
field.compute_potential(particles, alpha=1.0)

# Use origin='lower' so that array index 0 is displayed at Y=0
contour = ax_main.contourf(
    X, Y, field.potential,
    levels=100, cmap='viridis', alpha=0.7, origin='lower'
)

quiver = None

# Adam optimizer parameters
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
learning_rate = 5  # Base learning rate

# Adam variables
m = np.zeros_like(particles)
v = np.zeros_like(particles)
t = 0  # Time step

# Data storage for plots
coverage_data = []
max_coverage_data = []
single_coverage_data = []
double_coverage_data = []
excess_coverage_data = []
redundancy_data = []

cbar = None
current_text = None
num_text = None
optimal_text = None
circles = []

def update(frame):
    global particles, quiver, contour, cbar, m, v, t
    global current_text, num_text, optimal_text, circles

    # Add viewpoints after ADD_AT
    if frame in ADD_FRAMES:
        new_pts = np.random.rand(NEW_PARTICLES, 2) * grid_size
        particles = np.vstack([particles, new_pts])
        m = np.vstack([m, np.zeros_like(new_pts)])
        v = np.vstack([v, np.zeros_like(new_pts)])
        num_particles = particles.shape[0]
        field.repulsive_forces = np.zeros((num_particles, num_particles, 2))
        print(f"Frame {frame}: Added {NEW_PARTICLES} viewpoints")

    if frame in REMOVE_FRAMES and particles.shape[0] > NEW_PARTICLES:
        particles = particles[:-NEW_PARTICLES]
        m = m[:-NEW_PARTICLES]
        v = v[:-NEW_PARTICLES]
        num_particles = particles.shape[0]
        field.repulsive_forces = np.zeros((num_particles, num_particles, 2))
        print(f"Frame {frame}: Removed {NEW_PARTICLES} viewpoints")

    # 1) Update coverage and calculate metrics
    field.update_visibility(particles)
    coverage_metrics = field.compute_coverage_monte_carlo(particles, num_samples=10000, threshold=0.25)
    
    print(f"Frame {frame}: Coverage = {coverage_metrics['coverage'] * 100:.2f}%, "
          f"Max = {coverage_metrics['theoretical_max'] * 100:.2f}%, "
          f"Optimal Ratio = {coverage_metrics['optimal_ratio'] * 100:.2f}%, "
          f"Redundancy = {coverage_metrics['redundancy_ratio'] * 100:.2f}%")
    
    # Store metrics for plotting
    coverage_data.append(coverage_metrics['coverage'])
    max_coverage_data.append(coverage_metrics['theoretical_max'])
    single_coverage_data.append(coverage_metrics['single_coverage'])
    double_coverage_data.append(coverage_metrics['double_coverage'])
    excess_coverage_data.append(coverage_metrics['excess_coverage'])
    redundancy_data.append(coverage_metrics['redundancy_ratio'])

    # 2) Recompute potential (for visualization)
    field.compute_potential(particles, alpha=1.0)
    
    # 3) Compute forces
    total_forces = field.compute_force(particles, alpha=1.0, k_rep=0.05)

    # 4) Adam update for particles
    t += 1
    grad = total_forces
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    
    particles += learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    particles %= grid_size

    # Update visualization
    # Remove old contour
    for c in contour.collections:
        c.remove()
    # Re-draw with updated potential
    contour = ax_main.contourf(
        X, Y, field.potential,
        levels=100, cmap='viridis', alpha=1.0, origin='lower'
    )

    if cbar is not None:
        cbar.remove()
    # Create an axes divider for the colorbar positioning
    divider = make_axes_locatable(ax_main)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(contour, cax=cax)

    # Quiver showing the attractive force of the first particle only
    quiver_step = 5
    xq = X[::quiver_step, ::quiver_step]
    yq = Y[::quiver_step, ::quiver_step]
    if particles.shape[0] > 0:  # Make sure we have particles
        u_attr = field.attractive_forces[::quiver_step, ::quiver_step, 0, 0].flatten()
        v_attr = field.attractive_forces[::quiver_step, ::quiver_step, 0, 1].flatten()

        # Remove old quiver entirely, then recreate
        global quiver
        if quiver is not None:
            quiver.remove()
        quiver = ax_main.quiver(
            xq, yq, u_attr, v_attr,
            color="#ff1493", scale=1000, width=0.002, pivot="middle", zorder=99
        )

    # Update the particle scatter positions
    scatter.set_offsets(particles)
    
    # IMPORTANT FIX: Clean removal of all circles from previous frame
    for circle in circles:
        try:
            circle.remove()
        except:
            pass  # In case the circle was already removed
    
    # Clear the circles list
    circles = []
    
    # Create new circles for the current particle positions, handling toroidal boundaries
    for p in particles:
        # Original circle at the particle's position
        circle = plt.Circle((p[0], p[1]), field.fov_radius, fill=False, 
                         color='white', linestyle='-', linewidth=0.7, 
                         alpha=0.4, zorder=4)
        ax_main.add_patch(circle)
        circles.append(circle)
        
        # Check if the circle crosses boundaries and add wrapped portions if needed
        # The circle crosses a boundary if its distance to any edge is less than the radius
        
        # Check left boundary
        if p[0] < field.fov_radius:
            # Add a circle that wraps to the right side of the grid
            wrapped_circle = plt.Circle((p[0] + grid_size, p[1]), field.fov_radius, fill=False, 
                             color='white', linestyle='-', linewidth=0.7, 
                             alpha=0.4, zorder=4)
            ax_main.add_patch(wrapped_circle)
            circles.append(wrapped_circle)
        
        # Check right boundary
        if p[0] > grid_size - field.fov_radius:
            # Add a circle that wraps to the left side of the grid
            wrapped_circle = plt.Circle((p[0] - grid_size, p[1]), field.fov_radius, fill=False, 
                             color='white', linestyle='-', linewidth=0.7, 
                             alpha=0.4, zorder=4)
            ax_main.add_patch(wrapped_circle)
            circles.append(wrapped_circle)
        
        # Check bottom boundary
        if p[1] < field.fov_radius:
            # Add a circle that wraps to the top of the grid
            wrapped_circle = plt.Circle((p[0], p[1] + grid_size), field.fov_radius, fill=False, 
                             color='white', linestyle='-', linewidth=0.7, 
                             alpha=0.4, zorder=4)
            ax_main.add_patch(wrapped_circle)
            circles.append(wrapped_circle)
        
        # Check top boundary
        if p[1] > grid_size - field.fov_radius:
            # Add a circle that wraps to the bottom of the grid
            wrapped_circle = plt.Circle((p[0], p[1] - grid_size), field.fov_radius, fill=False, 
                             color='white', linestyle='-', linewidth=0.7, 
                             alpha=0.4, zorder=4)
            ax_main.add_patch(wrapped_circle)
            circles.append(wrapped_circle)
        
        # Add corner wrapping for particles near corners
        # Top-right corner
        if p[0] > grid_size - field.fov_radius and p[1] > grid_size - field.fov_radius:
            wrapped_circle = plt.Circle((p[0] - grid_size, p[1] - grid_size), field.fov_radius, fill=False, 
                             color='white', linestyle='-', linewidth=0.7, 
                             alpha=0.4, zorder=4)
            ax_main.add_patch(wrapped_circle)
            circles.append(wrapped_circle)
        
        # Top-left corner
        if p[0] < field.fov_radius and p[1] > grid_size - field.fov_radius:
            wrapped_circle = plt.Circle((p[0] + grid_size, p[1] - grid_size), field.fov_radius, fill=False, 
                             color='white', linestyle='-', linewidth=0.7, 
                             alpha=0.4, zorder=4)
            ax_main.add_patch(wrapped_circle)
            circles.append(wrapped_circle)
        
        # Bottom-right corner
        if p[0] > grid_size - field.fov_radius and p[1] < field.fov_radius:
            wrapped_circle = plt.Circle((p[0] - grid_size, p[1] + grid_size), field.fov_radius, fill=False, 
                             color='white', linestyle='-', linewidth=0.7, 
                             alpha=0.4, zorder=4)
            ax_main.add_patch(wrapped_circle)
            circles.append(wrapped_circle)
        
        # Bottom-left corner
        if p[0] < field.fov_radius and p[1] < field.fov_radius:
            wrapped_circle = plt.Circle((p[0] + grid_size, p[1] + grid_size), field.fov_radius, fill=False, 
                             color='white', linestyle='-', linewidth=0.7, 
                             alpha=0.4, zorder=4)
            ax_main.add_patch(wrapped_circle)
            circles.append(wrapped_circle)

    # Update the coverage line plot
    coverage_line.set_data(range(len(coverage_data)), coverage_data)
    max_coverage_line.set_data(range(len(max_coverage_data)), max_coverage_data)
    
    # Update the overlap metrics plot
    single_line.set_data(range(len(single_coverage_data)), single_coverage_data)
    double_line.set_data(range(len(double_coverage_data)), double_coverage_data)
    excess_line.set_data(range(len(excess_coverage_data)), excess_coverage_data)
    redundancy_line.set_data(range(len(redundancy_data)), redundancy_data)

    # Add marker only for the last data point on coverage plot
    coverage_line.set_marker('')
    if len(coverage_data) > 0:
        coverage_line.set_marker('o')
        coverage_line.set_markersize(4)
        coverage_line.set_markevery([len(coverage_data)-1])

    # Remove previous text (if it exists)
    if current_text is not None:
        current_text.remove()
    if num_text is not None:
        num_text.remove()
    if optimal_text is not None:
        optimal_text.remove()

    # Overlay the current coverage as text
    coverage_ratio = coverage_metrics['coverage'] / coverage_metrics['theoretical_max'] if coverage_metrics['theoretical_max'] > 0 else 0
    current_text = ax_line.text(
        0.95, 0.95, 
        f"Coverage: {coverage_metrics['coverage']*100:.2f}%\nMax: {coverage_metrics['theoretical_max']*100:.2f}%\nRatio: {coverage_ratio*100:.1f}%", 
        ha='right', va='top', 
        transform=ax_line.transAxes, 
        fontsize=8, 
        color='black'
    )
    
    # Add viewpoints count
    num_text = ax_main.text(
        0.02, 0.98, 
        f"Viewpoints: {particles.shape[0]}",
        ha='left', va='top',
        transform=ax_main.transAxes,
        fontsize=8,
        color='black'
    )
    
    # Add optimal ratio text
    optimal_text = ax_overlap.text(
        0.95, 0.95, 
        f"Optimal Ratio: {coverage_metrics['optimal_ratio']*100:.2f}%\nRedundancy: {coverage_metrics['redundancy_ratio']*100:.2f}%", 
        ha='right', va='top', 
        transform=ax_overlap.transAxes, 
        fontsize=8, 
        color='black'
    )

    # Adjust axis limits dynamically
    ax_line.set_xlim(0, max(len(coverage_data), frames))
    ax_overlap.set_xlim(0, max(len(coverage_data), frames))

    return scatter, contour, quiver, coverage_line, max_coverage_line, single_line, double_line, excess_line, redundancy_line

# Create animation
ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=False)

plt.tight_layout()

# Save animation
# ani.save('coverage_optimization.mp4', writer='ffmpeg', fps=30, dpi=150)

plt.show()

