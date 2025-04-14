import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

# Number of particles and grid size
grid_size = 100
frames = 200

# When to add/remove viewpoints (frame number)
REMOVE_FRAMES = []       # Frames to add viewpoints
ADD_FRAMES = []   # Frames to remove viewpoints
NEW_PARTICLES = 1  # Number of new viewpoints to add


# Initial positions of particles
num_particles = 6
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

        # Visibility array: 1 if covered, 0 otherwise
        self.visibility = np.zeros((grid_size, grid_size), dtype=np.float64)

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
        
        # Divergence field
        self.divergence = np.zeros((grid_size, grid_size))

    def compute_coverage_monte_carlo(self, num_samples=10000, sigma=10, threshold=0.0):
        """
        Computes coverage using Monte Carlo sampling.
        Instead of relying on a grid, it samples random points and checks their visibility.
        """
        # Sample random points in the space
        random_points = np.random.rand(num_samples, 2) * self.grid_size  # Shape: (num_samples, 2)

        # Compute visibility at sampled points
        diff = random_points[:, None, :] - particles[None, :, :]
        wrapped_diff = self.wrap_distance(diff)
        distances = np.linalg.norm(wrapped_diff, axis=-1) + epsilon

        # Visibility function
        visibility_values = (distances <= self.fov_radius).astype(float)
        # Aggregate visibility per sampled point
        sampled_visibility = np.clip(np.sum(visibility_values, axis=1), 0, 1)

        # Count how many sampled points exceed threshold
        covered_samples = np.sum(sampled_visibility > threshold)
        coverage_fraction = covered_samples / num_samples

        return coverage_fraction

    
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
        Mark each grid point as visible if it is within self.fov_radius
        of any particle. Otherwise 0.
        """
        self.visibility.fill(0)
        diff = self.field_points[:, None, :] - particles[None, :, :]
        wrapped_diff = self.wrap_distance(diff)
        distances = np.linalg.norm(wrapped_diff, axis=-1)

        # If a grid point is within fov_radius of at least one particle -> visible
        visibility_mask = np.any(distances <= self.fov_radius, axis=1)
        self.visibility.ravel()[visibility_mask] = 1


    def compute_potential(self, particles, alpha=1.0):
        """
        A log-based potential that is zero where visibility=1 (already covered),
        and grows with sum of log(distances) for uncovered areas.

        potential(point) = alpha * (1 - visibility(point))
                            * sum_{over particles}( log(distance + epsilon) )
        """
        diff = self.field_points[:, None, :] - particles[None, :, :]
        wrapped_diff = self.wrap_distance(diff)
        distances = np.linalg.norm(wrapped_diff, axis=-1) + epsilon

        # Sum of log(distance) across all particles
        log_sum = np.log(distances).sum(axis=1)  # (grid_size*grid_size,)

        # Zero potential if already covered: (1 - visibility)
        covered_factor = 1.0 - self.visibility.ravel()
        pot_values = alpha * covered_factor * log_sum

        self.potential = pot_values.reshape(self.grid_size, self.grid_size)

    def compute_attractive_force(self, particles, alpha=1.0):
        """
        Negative gradient of the coverage-based log potential.

        Force on each particle from each grid cell ~
          alpha * (1 - visibility) * sum_over_grid[ (1/distance) * direction ]
        """
        diff = self.field_points[:, None, :] - particles[None, :, :]
        wrapped_diff = self.wrap_distance(diff)
        distances = np.linalg.norm(wrapped_diff, axis=-1) + epsilon
        directions = wrapped_diff / distances[..., None]  # shape: (Npoints, Nparticles, 2)

        # (1 - visibility) zeroes out covered cells
        covered_factor = (1.0 - self.visibility.ravel())[:, None]  # shape: (Npoints, 1)

        # Combine factors: alpha * (1-vis) / distance
        coverage_need = alpha * covered_factor / distances
        coverage_need[distances < epsilon] = 0.0

        # Force vector from each grid point to each particle
        attractive_forces = coverage_need[..., None] * directions

        # Reshape to (grid_size, grid_size, num_particles, 2) for visualization
        self.attractive_forces = attractive_forces.reshape(
            self.grid_size, self.grid_size, particles.shape[0], 2
        )

        # Sum over all grid points => shape (num_particles, 2)
        total_attractive = self.attractive_forces.sum(axis=(0, 1))
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
                      k_attr=0.4, k_rep=0, alpha=1.0):
        """
        Combine coverage-based log attraction and Gaussian repulsion:
          F_total = k_attr * F_attr + k_rep * F_rep
        """
        total_attractive = self.compute_attractive_force(particles, alpha)
        total_repelling = self.compute_repelling_force(particles, sigma, amplitude)
        return k_attr * total_attractive + k_rep * total_repelling
    
    def compute_divergence(self, particles):
        """
        Compute the divergence of the vector field.
        The divergence tells us how much the field is expanding or contracting at each point.
        
        For a vector field F = [Fx, Fy], divergence = ∂Fx/∂x + ∂Fy/∂y
        """
        # First compute the attractive forces (if not already computed)
        self.compute_attractive_force(particles, alpha=1.0)
        
        # Sum over all particles to get the total field at each point
        # Shape of attractive_forces: (grid_size, grid_size, num_particles, 2)
        # After sum: (grid_size, grid_size, 2)
        total_field = np.sum(self.attractive_forces, axis=2)
        
        # Extract x and y components
        Fx = total_field[:, :, 0]
        Fy = total_field[:, :, 1]
        
        # Compute gradients using numpy's gradient function
        # gradient returns (∂f/∂y, ∂f/∂x) for a 2D input
        grad_x = np.gradient(Fx, axis=1)  # ∂Fx/∂x
        grad_y = np.gradient(Fy, axis=0)  # ∂Fy/∂y
        
        # Divergence = ∂Fx/∂x + ∂Fy/∂y
        self.divergence = grad_x + grad_y
        
        return self.divergence


# Initialize the field
field = Field(grid_size, num_particles, fov_radius=20)


# === FIGURE SETUP ===
fig = plt.figure(figsize=(18, 6), dpi=150)
gs = GridSpec(1, 3, width_ratios=[1, 1, 1])  # Three equal plots

ax_main = fig.add_subplot(gs[0, 0])
ax_div = fig.add_subplot(gs[0, 1])  # New subplot for divergence
ax_line = fig.add_subplot(gs[0, 2])
plt.subplots_adjust(wspace=0.3)  # Adjust the space between plots

# === LEFT PLOT: Field Visualization ===
ax_main.set_xlim(0, grid_size - 1)
ax_main.set_ylim(0, grid_size - 1)
ax_main.set_aspect("equal")
ax_main.set_title("Field Potential & Coverage", fontsize=12)

# === MIDDLE PLOT: Divergence Visualization ===
ax_div.set_xlim(0, grid_size - 1)
ax_div.set_ylim(0, grid_size - 1)
ax_div.set_aspect("equal")
ax_div.set_title("Field Divergence", fontsize=12)

# === RIGHT PLOT: Coverage Growth Over Time ===
ax_line.set_xlim(0, frames)
ax_line.set_ylim(0, 1)
ax_line.set_xlabel("Time Step", fontsize=10)
ax_line.set_ylabel("Coverage", fontsize=10)
ax_line.set_title("Coverage Over Time", fontsize=10)
ax_line.set_position([0.67, 0.225, 0.28, 0.55])  # [left, bottom, width, height]

# Coverage line with blue-purple gradient
coverage_line, = ax_line.plot([], [], color="#3d03fc")

# Initial computations
field.update_visibility(particles)
field.compute_potential(particles, alpha=1.0)
field.compute_divergence(particles)

# Contour plot with automatic colorbar for potential
contour = ax_main.contourf(
    X, Y, field.potential,
    levels=100, cmap='viridis', alpha=0.7, origin='lower'
)

# Divergence plot - use a diverging colormap
div_contour = ax_div.contourf(
    X, Y, field.divergence,
    levels=100, cmap='RdBu_r', alpha=0.7, origin='lower'
)

# Particle scatter plots
scatter = ax_main.scatter(
    particles[:, 0], particles[:, 1],
    c="deepskyblue", s=50, zorder=5
)

scatter_div = ax_div.scatter(
    particles[:, 0], particles[:, 1],
    c="deepskyblue", s=50, zorder=5
)

quiver = None
div_quiver = None  # For divergence plot
div_cbar = None    # For divergence colorbar

# Adam optimizer parameters
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
learning_rate = 5  # Base learning rate

# Adam variables
m = np.zeros_like(particles)
v = np.zeros_like(particles)
t = 0  # Time step

coverage_data = []
cbar = None
div_cbar = None
current_text = None
num_text = None
div_text = None

def update(frame):
    global particles, quiver, contour, cbar, m, v, t, current_text, num_text
    global div_contour, div_cbar, div_quiver, div_text

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

    # 1) Update coverage
    field.update_visibility(particles)
    coverage = field.compute_coverage_monte_carlo(num_samples=10000, threshold=0.25)
    print(f"Frame {frame}: Monte Carlo Coverage = {coverage * 100:.2f}%")
    coverage_data.append(coverage)

    # 2) Recompute potential (for visualization)
    field.compute_potential(particles, alpha=1.0)
    
    # 3) Compute forces
    total_forces = field.compute_force(particles, alpha=1.0)
    
    # 4) Compute divergence for the attraction field
    divergence = field.compute_divergence(particles)

    # 5) Adam update for particles
    t += 1
    grad = total_forces
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    
    particles += learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    particles %= grid_size

    # === UPDATE POTENTIAL PLOT ===
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
    cbar.set_label('Potential')

    # === UPDATE DIVERGENCE PLOT ===
    # Remove old divergence contour
    for c in div_contour.collections:
        c.remove()
    
    # Use a diverging colormap for divergence
    vmax = max(abs(np.min(divergence)), abs(np.max(divergence)))
    vmin = -vmax
    
    div_contour = ax_div.contourf(
        X, Y, divergence,
        levels=100, cmap='RdBu_r', alpha=1.0, origin='lower',
        vmin=vmin, vmax=vmax
    )

    if div_cbar is not None:
        div_cbar.remove()
    divider_div = make_axes_locatable(ax_div)
    cax_div = divider_div.append_axes("right", size="5%", pad=0.05)
    div_cbar = plt.colorbar(div_contour, cax=cax_div)
    div_cbar.set_label('Divergence')

    # Add divergence annotations
    # Find locations of high divergence (sources)
    threshold_high = vmax * 0.7
    source_locations = np.where(divergence > threshold_high)
    for i, j in zip(source_locations[0], source_locations[1]):
        if i % 5 == 0 and j % 5 == 0:  # Only plot every 5th point to avoid clutter
            ax_div.plot(j, i, 'ko', markersize=1, alpha=0.5)  # Mark sources with black dots
    
    # Find locations of low divergence (sinks)
    threshold_low = vmin * 0.7
    sink_locations = np.where(divergence < threshold_low)
    for i, j in zip(sink_locations[0], sink_locations[1]):
        if i % 5 == 0 and j % 5 == 0:  # Only plot every 5th point to avoid clutter
            ax_div.plot(j, i, 'wo', markersize=1, alpha=0.5)  # Mark sinks with white dots

    # === VECTOR FIELD VISUALIZATION ===
    # Quiver showing the attractive force of the first particle only, for demonstration
    quiver_step = 5
    xq = X[::quiver_step, ::quiver_step]
    yq = Y[::quiver_step, ::quiver_step]
    # shape (grid_size, grid_size, num_particles, 2)
    u_attr = field.attractive_forces[::quiver_step, ::quiver_step, 0, 0].flatten()
    v_attr = field.attractive_forces[::quiver_step, ::quiver_step, 0, 1].flatten()

    # Remove old quiver entirely, then recreate
    if quiver is not None:
        quiver.remove()
    quiver = ax_main.quiver(
        xq, yq, u_attr, v_attr,
        color="#ff1493", scale=1000, width=0.002, pivot="middle", zorder=99
    )

    # === UPDATE PARTICLE POSITIONS ===
    scatter.set_offsets(particles)
    scatter_div.set_offsets(particles)

    # === UPDATE COVERAGE PLOT ===
    coverage_line.set_data(range(len(coverage_data)), coverage_data)

    # Add marker only for the last data point
    coverage_line.set_marker('')
    if len(coverage_data) > 0:
        coverage_line.set_marker('o')  # Set marker for the last point
        coverage_line.set_markersize(4)  # Set the size of the marker
        coverage_line.set_markevery([len(coverage_data)-1])  # Only show marker for the last point

    # === UPDATE TEXT ELEMENTS ===
    # Remove previous text (if it exists)
    if current_text is not None:
        current_text.remove()

    # Overlay the current coverage as text
    current_text = ax_line.text(0.95, 0.95, f"Coverage: {coverage*100:.2f}%", ha='right', va='top', 
                 transform=ax_line.transAxes, fontsize=8, color='black',)
    
    if num_text is not None:
        num_text.remove()

    num_text = ax_main.text(
        -0.25, -0.25, f"Viewpoints: {particles.shape[0]}",
        ha='left', va='top',
        transform=ax_main.transAxes,
        fontsize=8,
    )

    # Add divergence explanation text
    if div_text is not None:
        div_text.remove()
    
    div_min = np.min(divergence)
    div_max = np.max(divergence)
    div_text = ax_div.text(
        0.05, 0.05, 
        f"Min: {div_min:.2e}\nMax: {div_max:.2e}\n\nRed = Sources\nBlue = Sinks",
        ha='left', va='bottom',
        transform=ax_div.transAxes,
        fontsize=8, bbox=dict(facecolor='white', alpha=0.7)
    )

    ax_line.set_xlim(0, max(len(coverage_data), frames))  # Adjust X-axis dynamically

    return scatter, contour, quiver, coverage_line, div_contour

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=False)

plt.tight_layout()
plt.show()

# Save the animation with a descriptive name
ani.save('field_divergence_visualization.mp4', writer='ffmpeg', fps=24, dpi=300)