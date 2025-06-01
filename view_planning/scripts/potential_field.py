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
num_particles = 11
# particles = np.random.rand(num_particles, 2) * grid_size

particles = 50 + np.random.rand(num_particles, 2) 
# particles = np.zeros((num_particles, 2))
# Create a mesh grid for the potential field
# Note: X, Y will be shaped (grid_size, grid_size).
# X[i,j] = j, Y[i,j] = i by default for np.meshgrid(x, y).
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

        # Gaussian-based visibility function
        # visibility_values = np.exp(-distances**2 / (2 * sigma**2))  # Using sigma=10
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

    # def update_visibility(self, particles, sigma=10):
    #     """
    #     Compute Gaussian visibility instead of a binary 0 or 1 visibility.
    #     Visibility will smoothly transition from 1 near a particle to 0 far away.
    #     """
    #     self.visibility.fill(0)
    #     diff = self.field_points[:, None, :] - particles[None, :, :]
    #     wrapped_diff = self.wrap_distance(diff)
    #     distances = np.linalg.norm(wrapped_diff, axis=-1) + epsilon  # Avoid div by 0

    #     # Gaussian-based visibility function
    #     gaussian_visibility = np.exp(-distances**2 / (2 * sigma**2))

    #     # Take the maximum visibility contribution from all particles
    #     # self.visibility.ravel()[:] = np.clip(np.sum(gaussian_visibility, axis=1), 0, 1)
    #     self.visibility.ravel()[:] = np.max(gaussian_visibility, axis=1)


    # def update_visibility(self, particles):
    #     """
    #     Mark each grid point as visible if it is within self.fov_radius
    #     of any particle. Otherwise 0.
    #     """
    #     self.visibility.fill(0)
    #     diff = self.field_points[:, None, :] - particles[None, :, :]
    #     wrapped_diff = self.wrap_distance(diff)
    #     distances = np.linalg.norm(wrapped_diff, axis=-1)

    #     # If a grid point is within fov_radius of at least one particle -> visible
    #     visibility_mask = np.any(distances <= self.fov_radius, axis=1)
    #     self.visibility.ravel()[visibility_mask] = 1

    def update_visibility(self, particles, beta=20.0):
        """
        Soft visibility using a steep sigmoid approximation to a hard FOV:
            s(r) = 1 / (1 + exp[β (r / R - 1)])
        Where:
            - R is the FOV radius
            - β controls sharpness (higher = closer to binary step)
        """
        self.visibility.fill(0)
        diff = self.field_points[:, None, :] - particles[None, :, :]
        wrapped_diff = self.wrap_distance(diff)
        distances = np.linalg.norm(wrapped_diff, axis=-1) + epsilon  # (Npoints, Nparticles)

        s = 1.0 / (1.0 + np.exp(beta * (distances / self.fov_radius - 1.0)))  # soft visibility
        self.visibility.ravel()[:] = np.clip(np.sum(s, axis=1), 0.0, 1.0)  # values in [0,1]



    def compute_potential(self, particles, alpha=1.0):
        """
        A log-based potential that is zero where visibility=1 (already covered),
        and grows with sum of log(distances) for uncovered areas.

        potential(point) = alpha * (1 - visibility(point))
                            * sum_{over particles}( log(distance + epsilon) )

        This is just for visualization. The gradient for the force will use
        the same logic in compute_attractive_force.
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

        We effectively sum alpha*(1 - vis)*log(distance) across grid cells
        and take the gradient w.r.t. particle positions.
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


    # def compute_potential(self, particles, alpha=1.0, r0=1.0):
    #     """
    #     Smoothed log-based potential:
    #         φ(x) = 1/2 * log(1 + ||x - p||^2 / r0^2)
    #     for each viewpoint p, summed over all particles.
    #     Only contributes where visibility is 0.
    #     """
    #     diff = self.field_points[:, None, :] - particles[None, :, :]
    #     wrapped_diff = self.wrap_distance(diff)
    #     squared_distances = np.sum(wrapped_diff ** 2, axis=-1)  # Shape: (Npoints, Nparticles)

    #     log_sum = 0.5 * np.log1p(squared_distances / (r0 ** 2)).sum(axis=1)

    #     covered_factor = 1.0 - self.visibility.ravel()
    #     pot_values = alpha * covered_factor * log_sum
    #     self.potential = pot_values.reshape(self.grid_size, self.grid_size)


    # def compute_attractive_force(self, particles, alpha=1.0, r0=1.0):
    #     """
    #     Gradient of the log-smooth potential:
    #         ∇φ(x) ∝ (x - p) / (r0^2 + ||x - p||^2)
    #     Weighted by coverage deficit: (1 - visibility).
    #     """
    #     diff = self.field_points[:, None, :] - particles[None, :, :]
    #     wrapped_diff = self.wrap_distance(diff)
    #     squared_distances = np.sum(wrapped_diff ** 2, axis=-1) + epsilon
    #     denom = r0 ** 2 + squared_distances  # Shape: (Npoints, Nparticles)

    #     covered_factor = (1.0 - self.visibility.ravel())[:, None]  # Shape: (Npoints, 1)
    #     force_magnitudes = alpha * covered_factor / denom  # Shape: (Npoints, Nparticles)

    #     attractive_forces = force_magnitudes[..., None] * wrapped_diff  # Shape: (Npoints, Nparticles, 2)

    #     self.attractive_forces = attractive_forces.reshape(
    #         self.grid_size, self.grid_size, particles.shape[0], 2
    #     )

    #     total_attractive = self.attractive_forces.sum(axis=(0, 1))
    #     return total_attractive


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


# Initialize the field
field = Field(grid_size, num_particles, fov_radius=20)


# === FIGURE SETUP ===
# fig, (ax_main, ax_line) = plt.subplots(1, 2, figsize=(14, 6), dpi=150, )

fig = plt.figure(figsize=(14, 6), dpi=150)
gs = GridSpec(1, 2, width_ratios=[1, 1])  # Equal space allocation for both plots

ax_main = fig.add_subplot(gs[0, 0])
ax_line = fig.add_subplot(gs[0, 1])
plt.subplots_adjust(wspace=0.5)  # Adjust the space between plots (increase the value)

# === LEFT PLOT: Field Visualization ===
ax_main.set_xlim(0, grid_size - 1)
ax_main.set_ylim(0, grid_size - 1)
# ax_main.set_aspect("equal")  # Ensures square aspect ratio
ax_main.set_title("Field Potential & Coverage", fontsize=12)
ax_main.set_aspect("equal")  # This allows the layout to adjust naturally


# Contour plot with automatic colorbar
contour = ax_main.contourf(
    X, Y, field.potential, 
    levels=100, cmap="viridis", alpha=0.9, origin="lower"
)

# Particle styling
scatter = ax_main.scatter(
    particles[:, 0], particles[:, 1],
    c="deepskyblue", s=50,  zorder=5
)


# === RIGHT PLOT: Coverage Growth Over Time ===
ax_line.set_xlim(0, frames)
ax_line.set_ylim(0, 1)
# ax_line.set_box_aspect(1)  # **NEW FIX: Prevents aspect ratio distortion**
ax_line.set_xlabel("Time Step", fontsize=10)
ax_line.set_ylabel("Coverage", fontsize=10)
ax_line.set_title("Coverage Over Time", fontsize=10)
# ax_line.set_aspect('auto')  # Let it adjust naturally as well
ax_line.set_position([0.53, 0.225, 0.4, 0.55])  # [left, bottom, width, height]

# Coverage line with blue-purple gradient
coverage_line, = ax_line.plot([], [], color="#3d03fc" )



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

coverage_data = []
cbar = None
current_text = None
num_text = None

def update(frame):
    global particles, quiver, contour, cbar, m, v, t, current_text, num_text

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
    total_forces = field.compute_force(particles, alpha=1.0, k_attr=0.4, k_rep=0.1)


    # # 4) Move particles with toroidal wrapping
    # particles += total_forces
    # particles %= grid_size

    # 4) Adam update for particles
    t += 1
    grad = total_forces
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    
    particles += learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    particles %= grid_size

    # Remove old contour
    for c in contour.collections:
        c.remove()
    # Re-draw with updated potential
    # Use origin='lower' to keep array row 0 near the bottom
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


    # Quiver showing the attractive force of the first particle only, for demonstration
    quiver_step = 5
    xq = X[::quiver_step, ::quiver_step]
    yq = Y[::quiver_step, ::quiver_step]
    # shape (grid_size, grid_size, num_particles, 2)
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

    # Update the line plot
    coverage_line.set_data(range(len(coverage_data)), coverage_data)

    # Add marker only for the last data point
    coverage_line.set_marker('')
    if len(coverage_data) > 0:
        coverage_line.set_marker('o')  # Set marker for the last point
        coverage_line.set_markersize(4)  # Set the size of the marker

        coverage_line.set_markevery([len(coverage_data)-1])  # Only show marker for the last point

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

    ax_line.set_xlim(0, max(len(coverage_data), frames))  # Adjust X-axis dynamically
  

    return scatter, contour, quiver, coverage_line



ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=False)

plt.legend(loc='upper right')

plt.show()

ani.save('gaussian 6.mp4', writer='ffmpeg', fps=24, dpi=300)

