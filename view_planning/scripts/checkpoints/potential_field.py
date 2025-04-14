import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Number of particles and grid size
grid_size = 100
frames = 120

# Generate random particle positions (x, y)
num_particles = 3
# particles = np.random.rand(num_particles, 2) * grid_size
particles = np.array([[20, 20], [30, 20], [80, 20]], dtype=np.float64)
# Create a mesh grid for the potential field
x = np.arange(grid_size)
y = np.arange(grid_size)
X, Y = np.meshgrid(x, y)
field_points = np.stack([X.ravel(), Y.ravel()], axis=-1)

epsilon = 1e-6  # Avoid division by zero

# Field Class with Optimized Strategies
# Field Class with Optimized Strategies
class Field:
    def __init__(self, grid_size, num_particles, fov_radius, use_wrapping=True):
        self.grid_size = grid_size
        self.fov_radius = fov_radius
        self.use_wrapping = use_wrapping
        self.visibility = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.potential = np.zeros((grid_size, grid_size))
        self.field_points = field_points

        # Attractive forces remain a field grid (grid_size x grid_size x 2 x num_particles)
        self.attractive_forces = np.zeros((grid_size, grid_size, 2, num_particles), dtype=np.float64)

        # **Corrected: Repulsive forces are a `num_particles x num_particles x 2` matrix**
        # This matrix stores the repulsive force from particle `j` acting on particle `i`
        self.repulsive_forces = np.zeros((num_particles, num_particles, 2), dtype=np.float64)

    def wrap_distance(self, diff):
        """
        Compute wrapped distance for toroidal boundary conditions.
        """
        if not self.use_wrapping:
            return diff
        return np.where(np.abs(diff) > self.grid_size / 2, -np.sign(diff) * (self.grid_size - np.abs(diff)), diff)

    def update_visibility(self, particles):
        """
        Optimized visibility update using vectorized operations.
        """
        self.visibility.fill(0)
        diff = self.field_points[:, None, :] - particles[None, :, :]
        wrapped_diff = self.wrap_distance(diff)
        distances = np.linalg.norm(wrapped_diff, axis=-1)
        visibility_mask = np.any(distances <= self.fov_radius, axis=1)
        self.visibility.ravel()[visibility_mask] = 1

    def compute_potential(self, particles, sigma=10):
        """
        Optimized Gaussian potential computation.
        """
        diff = self.field_points[:, None, :] - particles[None, :, :]
        wrapped_diff = self.wrap_distance(diff)
        distances = np.linalg.norm(wrapped_diff, axis=-1)

        # Sum the Gaussian contributions
        gaussian_contributions = np.exp(-distances ** 2 / (2 * sigma ** 2)).sum(axis=1)

        # Assign correctly to potential without ravel()
        self.potential = (1 - gaussian_contributions).reshape(self.potential.shape)

    def compute_force(self, particles, sigma=10, amplitude=100, k_attr=0.006, k_rep=0.0):
        """
        Compute both attractive forces (field-based) and repulsive forces (particle-based).
        """
        forces = np.zeros_like(particles)
        self.attractive_forces.fill(0)
        self.repulsive_forces.fill(0)  # Reset for each frame

        # **Compute Attractive Forces (Gaussian Field)**
        diff = self.field_points[:, None, :] - particles[None, :, :]
        wrapped_diff = self.wrap_distance(diff)
        distances = np.linalg.norm(wrapped_diff, axis=-1) + epsilon
        directions = wrapped_diff / distances[..., None]  # Shape (10000, num_particles, 2)

        deprivation_potential = 1 - np.exp(-distances ** 2 / (2 * sigma ** 2)).sum(axis=1)
        attractive_forces = ((1 - self.visibility.ravel())[:, None, None] 
                            * deprivation_potential[:, None, None] 
                            * directions)

        # Reshape into (grid_size, grid_size, 2, num_particles)
        self.attractive_forces = attractive_forces.reshape(self.grid_size, self.grid_size, 2, num_particles)

        # Compute total attractive force acting on each particle (sum over all grid points)
        total_attractive = self.attractive_forces.sum(axis=(0, 1))  # Shape: (num_particles, 2)

        # **Compute Particle-to-Particle Repulsion (Correct Matrix Form)**
        pairwise_diff = particles[:, None, :] - particles[None, :, :]
        wrapped_pairwise_diff = self.wrap_distance(pairwise_diff)
        pairwise_distances = np.linalg.norm(wrapped_pairwise_diff, axis=-1) + epsilon

        repelling_forces = -(
            amplitude * wrapped_pairwise_diff
            * (-pairwise_distances[..., None] / sigma**2)
            * np.exp(-pairwise_distances**2 / (2 * sigma**2))[..., None]
        )

        # **Store in `num_particles x num_particles x 2` matrix**
        self.repulsive_forces = repelling_forces  # Force from `j` on `i`

        # Compute total repelling force acting on each particle (sum over columns)
        total_repelling = self.repulsive_forces.sum(axis=1)  # Shape: (num_particles, 2)

        # ✅ Ensure `total_attractive` has shape (num_particles, 2)
        total_attractive = total_attractive.reshape(num_particles, 2)

        # **Sum up forces affecting each particle**
        forces = k_attr * total_attractive + k_rep * total_repelling

        return forces






# Initialize the field with optimized processing
field = Field(grid_size, num_particles, fov_radius=10)

# Visualization and Animation
fig, ax = plt.subplots(figsize=(12, 9))
ax.set_xlim(0, grid_size - 1)
ax.set_ylim(0, grid_size - 1)
ax.set_aspect('equal')
ax.set_title("Potential Field with Forces")

scatter = ax.scatter(particles[:, 0], particles[:, 1], c='blue', s=50, zorder=5, label='Particles')
field.update_visibility(particles)
field.compute_potential(particles)
contour = ax.contourf(X, Y, field.potential, levels=50, cmap='plasma', alpha=0.7)

quiver = None


def update(frame):
    global particles, quiver, contour, field

    # Update visibility and potential
    field.update_visibility(particles)
    field.compute_potential(particles)

    # Compute total forces acting on particles
    total_forces = field.compute_force(particles)

    # ✅ Apply forces to ALL particles, not just the first one
    particles[0,:] += total_forces[0, :]  # Move all particles
    particles %= grid_size  # Wrap positions for toroidal boundaries

    # === Fix: Remove Previous Contour Plot ===
    for c in contour.collections:
        c.remove()
    contour = ax.contourf(X, Y, field.potential, levels=50, cmap='plasma', alpha=1.0)

    # === Update Quiver Plot ===
    quiver_step = 5  # Reduce quiver density for clarity
    x = X[::quiver_step, ::quiver_step]
    y = Y[::quiver_step, ::quiver_step]

    # ✅ Extract forces at each field point **for the first particle only**
    u_attr = field.attractive_forces[::quiver_step, ::quiver_step, 0, 0].flatten()
    v_attr = field.attractive_forces[::quiver_step, ::quiver_step, 1, 0].flatten()

    # ✅ Correct repulsion indexing: Sum forces from all other particles acting on particle 0
    u_rep = 0#field.repulsive_forces[0, :, 0].sum()  # Sum over all repelling particles
    v_rep = 0#field.repulsive_forces[0, :, 1].sum()  # Sum over all repelling particles

    # Ensure quiver is initialized before updating
    global quiver_initialized
    if quiver is None or not quiver_initialized:
        quiver = ax.quiver(
            x, y, u_attr + u_rep, v_attr + v_rep,
            color="#ff1493", scale=15, width=0.002, pivot="middle", zorder=99
        )
        quiver_initialized = True
    else:
        quiver.set_UVC(u_attr + u_rep, v_attr + v_rep)

    # === Update Scatter Plot ===
    scatter.set_offsets(particles)

    return scatter, quiver, contour












ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=False)

plt.colorbar(contour)
plt.legend(loc='upper right')
plt.show()

ani.save('force_with_quiver.mp4', writer='ffmpeg', fps=30, dpi=300)
