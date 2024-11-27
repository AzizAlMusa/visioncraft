import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Number of particles and grid size
num_particles = 2
grid_size = 100
frames = 180
k_repel = 100  # Repelling force constant

# Generate random particle positions (x, y)
particles = np.random.rand(num_particles, 2) * grid_size
particle1 = np.array([20, 20], dtype=float)
particle2 = np.array([20, 25], dtype=float)
particles = np.stack([particle1, particle2], axis=0)
# Create a mesh grid for the potential field
x = np.linspace(0, grid_size-1, grid_size)
y = np.linspace(0, grid_size-1, grid_size)
X, Y = np.meshgrid(x, y)

# Define a small epsilon to avoid division by zero
epsilon = 1e-6

# Field class to handle potential, visibility, and force computations
class Field:
    def __init__(self, grid_size, fov_radius, log=False):
        self.grid_size = grid_size
        self.fov_radius = fov_radius
        self.log = log
        self.visibility = np.ones((grid_size, grid_size), dtype=bool)
        self.potential = np.zeros((grid_size, grid_size))
        self.field_points = np.stack([X.ravel(), Y.ravel()], axis=-1)
        self.forces = np.zeros((grid_size, grid_size, 2))  # To store force vectors for quiver

    def reset_visibility(self):
        """Reset the visibility field to all points visible."""
        self.visibility.fill(0)

    def update_visibility(self, particles):
        """Update visibility field based on particles' positions."""
        for particle in particles:
            distances = np.linalg.norm(self.field_points - particle, axis=1)
            inside_fov = distances <= self.fov_radius
            self.visibility.ravel()[inside_fov] = 1  # Mark points in FOV as not visible

    def compute_potential(self, particles):
        """Compute potential field based on visibility and particles."""
        self.potential.fill(100)  # Reset potential to a high value
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # if not self.visibility[i, j]:  # Skip points inside FOV
                #     self.potential[i, j] = 0
                # else:
                visibility_function = (1 - self.visibility[i, j])
                distances = np.linalg.norm(particles - np.array([j, i]), axis=1)
                self.potential[i, j] = visibility_function * (
                    np.log(np.sum(distances)) if self.log else np.sum(distances)
                )

    def compute_force(self, particles):
        """Compute forces on particles based on the potential field."""
        forces = np.zeros_like(particles)
        self.forces.fill(0)  # Reset stored forces for visualization
        for idx, particle in enumerate(particles):
            total_force = np.zeros(2)
            for field_point, visibility in zip(self.field_points, self.visibility.ravel()):
                visibility_function = 1 - visibility  # Visible points contribute no force
                distance = np.linalg.norm(field_point - particle) + epsilon
                direction = (field_point - particle) / distance  # Unit vector
                force = visibility_function * 1 / distance * direction
                total_force += force

                # Store force vectors for the first particle for visualization
                if idx == 0:  # Only store forces for the first particle
                    i, j = int(field_point[1]), int(field_point[0])
                    self.forces[i, j] = force
            forces[idx] = total_force  # Accumulate total force
        return forces

    def compute_repelling_force(self, particles):
        """Compute repelling forces between particles."""
        repelling_forces = np.zeros_like(particles)
        for i in range(len(particles)):
            for j in range(len(particles)):
                if i == j:
                    continue
                distance = np.linalg.norm(particles[i] - particles[j]) + epsilon
                direction = (particles[i] - particles[j]) / distance  # Unit vector away from other particle
                repelling_force = k_repel / distance * direction
                repelling_forces[i] += repelling_force
        return repelling_forces

# Initialize the field
fov_radius = 30
field = Field(grid_size, fov_radius, log=False)

# Create the figure and axis for plotting
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, grid_size-1)
ax.set_ylim(0, grid_size-1)
ax.set_aspect('equal')
ax.set_title("Potential Field with Forces")

# Plot the particles
scatter = ax.scatter(particles[:, 0], particles[:, 1], c='blue', s=50, zorder=5, label='Particles')

# Initialize the potential field plot
field.update_visibility(particles)
field.compute_potential(particles)
contour = ax.contourf(X, Y, field.potential, levels=50, cmap='plasma', alpha=0.7)

# Quiver plot for visualizing forces
quiver = None

# Function to update the plot for each frame of the animation
def update(frame):
    global particles, quiver

    # Reset and update visibility
    field.reset_visibility()
    field.update_visibility(particles)

    # Recompute potential field
    field.compute_potential(particles)

    # Compute field-based forces
    field_forces = field.compute_force(particles)
    print("Attractive Forces:")
    print(field_forces)
    print("====================================")
    # Compute repelling forces
    repelling_forces = field.compute_repelling_force(particles)

    # Combine forces
    total_forces = 0.1 * field_forces + repelling_forces
    particles += total_forces * 0.5  # Apply a scaling factor for movement speed

    # Ensure particles stay within bounds
    particles = np.clip(particles, 0, grid_size-1)

    # Update the contour plot for potential field
    ax.contourf(X, Y, field.potential, levels=50, cmap='plasma', alpha=0.7)

    # Update the quiver plot
    if quiver:
        quiver.remove()
    quiver_step = 5  # Step size to reduce the number of quiver vectors
    x = X[::quiver_step, ::quiver_step]
    y = Y[::quiver_step, ::quiver_step]
    u = field.forces[::quiver_step, ::quiver_step, 0]
    v = field.forces[::quiver_step, ::quiver_step, 1]
    quiver = ax.quiver(x, y, u, v, color='magenta', scale=2, pivot='middle')

    # Update the scatter plot for particles
    scatter.set_offsets(particles)

    # Debugging: print particle positions
    print(f"Frame {frame}: Particle Positions: {particles}")

    return scatter, quiver

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=False)

# Display the animation
plt.colorbar(contour)
plt.legend(loc='upper right')
plt.show()

# ani.save('force_with_quiver.mp4', writer='ffmpeg', fps=60)
