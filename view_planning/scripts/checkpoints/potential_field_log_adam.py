import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Number of particles and grid size
grid_size = 100
frames = 120


# Initial positions of particles
num_particles = 4
particles = np.random.rand(num_particles, 2) * grid_size


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
        Combine coverage-based log attraction and Gaussian repulsion:
          F_total = k_attr * F_attr + k_rep * F_rep
        """
        total_attractive = self.compute_attractive_force(particles, alpha)
        total_repelling = self.compute_repelling_force(particles, sigma, amplitude)
        return k_attr * total_attractive + k_rep * total_repelling


# Initialize the field
field = Field(grid_size, num_particles, fov_radius=20)

# Set up figure
fig, ax = plt.subplots(figsize=(12, 9))
ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.set_aspect('equal')
ax.set_title("Log Potential + Coverage + Gaussian Repulsion")

# Scatter for particle positions
scatter = ax.scatter(particles[:, 0], particles[:, 1],
                     c='blue', s=50, zorder=5, label='Particles')

# Initial computations
field.update_visibility(particles)
field.compute_potential(particles, alpha=1.0)

# Use origin='lower' so that array index 0 is displayed at Y=0
contour = ax.contourf(
    X, Y, field.potential,
    levels=50, cmap='plasma', alpha=0.7, origin='lower'
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

def update(frame):
    global particles, quiver, contour, m, v, t

    # 1) Update coverage
    field.update_visibility(particles)
    # 2) Recompute potential (for visualization)
    field.compute_potential(particles, alpha=1.0)
    # 3) Compute forces
    total_forces = field.compute_force(particles, alpha=1.0)

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
    contour = ax.contourf(
        X, Y, field.potential,
        levels=50, cmap='plasma', alpha=1.0, origin='lower'
    )

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
    quiver = ax.quiver(
        xq, yq, u_attr, v_attr,
        color="#ff1493", scale=1, width=0.002, pivot="middle", zorder=99
    )

    # Update the particle scatter positions
    scatter.set_offsets(particles)

    return scatter, contour, quiver



ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=False)

plt.colorbar(contour, ax=ax)
plt.legend(loc='upper right')
plt.show()

ani.save('log_coverage_potential_fixed.mp4', writer='ffmpeg', fps=30, dpi=300)
