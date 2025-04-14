import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Set high-quality plotting defaults
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.minor.width'] = 0.8
plt.rcParams['ytick.minor.width'] = 0.8
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Custom color maps for beautiful visualization
# High contrast colormap for coverage
blue_cream = LinearSegmentedColormap.from_list('blue_cream', ['#0a306b', '#95d4ff', '#fffce6'])

# Potential colormap based on your style
potential_cmap = LinearSegmentedColormap.from_list('potential', ['#120638', '#561d7e', '#b63e81', '#ed8a65', '#fede8b'])

# Discrete colormap for particles
particle_colors = ['#fc4768', '#7a76fc', '#18c996', '#f1ca3a', '#3ab1fc', '#fa8c56']

# Configuration
grid_size = 100
num_viewpoints = 8  # Increased for better coverage
num_swarm_particles = 10  # Number of PSO particles (each containing a full solution)
frames = 200
fov_radius = 17  # Field of view radius

# Prepare grid for visualization
x = np.arange(grid_size)
y = np.arange(grid_size)
X, Y = np.meshgrid(x, y)
field_points = np.stack([X.ravel(), Y.ravel()], axis=-1)
epsilon = 1e-6

class Field:
    """This class handles the coverage calculation and visualization"""
    def __init__(self, grid_size, fov_radius, use_wrapping=True):
        self.grid_size = grid_size
        self.fov_radius = fov_radius
        self.use_wrapping = use_wrapping
        self.visibility = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.field_points = field_points
        
        # For force field visualization
        self.attractive_forces = None
        self.potential = None

    def wrap_distance(self, diff):
        """Compute toroidal wrapping so distance remains within [-grid_size/2, grid_size/2]."""
        if not self.use_wrapping:
            return diff
        return np.where(
            np.abs(diff) > self.grid_size / 2,
            -np.sign(diff) * (self.grid_size - np.abs(diff)),
            diff
        )

    def update_visibility(self, particles):
        """Mark each grid point as visible if it is within fov_radius of any particle."""
        self.visibility.fill(0)
        diff = self.field_points[:, None, :] - particles[None, :, :]
        wrapped_diff = self.wrap_distance(diff)
        distances = np.linalg.norm(wrapped_diff, axis=-1)
        visibility_mask = np.any(distances <= self.fov_radius, axis=1)
        self.visibility.ravel()[visibility_mask] = 1
    
    def compute_potential(self, particles, alpha=1.0):
        """
        Compute inverse potential field: Φ(r) ∝ 1/r
        """
        # Initialize a grid for potential
        potential = np.zeros((self.grid_size, self.grid_size))
        
        # Compute distances from every grid point to every particle
        diff = self.field_points[:, None, :] - particles[None, :, :]
        wrapped_diff = self.wrap_distance(diff)
        distances = np.linalg.norm(wrapped_diff, axis=-1) + epsilon
        
        # Inverse potential
        inv_sum = (1.0 / distances).sum(axis=1)
        
        # Scale by coverage mask
        coverage_factor = 1.0 - self.visibility.ravel()
        pot_values = alpha * coverage_factor * inv_sum
        
        self.potential = pot_values.reshape(self.grid_size, self.grid_size)
        return self.potential
    
    def compute_forces(self, particles, alpha=1.0):
        """
        Compute force field for visualization (negative gradient of potential)
        Returns: attractive_forces (grid_size, grid_size, 2)
        """
        # Force field on a sparser grid for visualization
        step = 5
        sparse_grid_size = self.grid_size // step
        forces = np.zeros((sparse_grid_size, sparse_grid_size, 2))
        
        # Create sparser grid for force computation
        sparse_x = np.linspace(0, self.grid_size-1, sparse_grid_size)
        sparse_y = np.linspace(0, self.grid_size-1, sparse_grid_size)
        SX, SY = np.meshgrid(sparse_x, sparse_y)
        sparse_points = np.stack([SX.ravel(), SY.ravel()], axis=-1)
        
        # Compute forces
        for i, point in enumerate(sparse_points):
            # Get distances to all particles
            diffs = particles - point
            wrapped_diffs = self.wrap_distance(diffs)
            distances = np.linalg.norm(wrapped_diffs, axis=1) + epsilon
            
            # Compute force vectors (inverse squared law)
            force_magnitudes = alpha / (distances**2)
            unit_vectors = wrapped_diffs / distances[:, None]
            forces_from_particles = force_magnitudes[:, None] * unit_vectors
            
            # Sum forces from all particles
            total_force = forces_from_particles.sum(axis=0)
            
            # Store in grid
            grid_i = i // sparse_grid_size
            grid_j = i % sparse_grid_size
            forces[grid_i, grid_j] = total_force
            
        self.attractive_forces = forces
        return forces, SX, SY
        
    def compute_coverage_monte_carlo(self, particles, num_samples=10000, threshold=0.0):
        """Computes coverage using Monte Carlo sampling."""
        # Sample random points in the space
        random_points = np.random.rand(num_samples, 2) * self.grid_size

        # Compute visibility at sampled points
        diff = random_points[:, None, :] - particles[None, :, :]
        wrapped_diff = self.wrap_distance(diff)
        distances = np.linalg.norm(wrapped_diff, axis=-1) + epsilon
        visibility_values = (distances <= self.fov_radius).astype(float)
        
        # Aggregate visibility per sampled point
        sampled_visibility = np.clip(np.sum(visibility_values, axis=1), 0, 1)

        # Count how many sampled points exceed threshold
        covered_samples = np.sum(sampled_visibility > threshold)
        coverage_fraction = covered_samples / num_samples

        return coverage_fraction


class PSOParticle:
    """
    A single particle in the PSO swarm.
    Each particle represents a potential solution (a complete configuration of all viewport positions).
    """
    def __init__(self, grid_size, num_viewpoints):
        # Position represents a full solution (all viewport positions)
        # Shape: (num_viewpoints, 2)
        self.position = np.random.rand(num_viewpoints, 2) * grid_size
        
        # Velocity for the particle (for all viewports together)
        # Shape: (num_viewpoints, 2)
        self.velocity = np.random.uniform(-2, 2, (num_viewpoints, 2))
        
        # Personal best position and fitness
        self.best_position = self.position.copy()
        self.best_fitness = -np.inf  # We're maximizing coverage
        
        # Current fitness
        self.fitness = -np.inf


class PSO:
    """Particle Swarm Optimization implementation for coverage maximization"""
    def __init__(self, field, num_viewpoints, num_swarm_particles, grid_size,
                 w=0.7, c1=1.4, c2=1.4):
        """
        Initialize the PSO algorithm
        
        Parameters:
        - field: Field object for computing coverage
        - num_viewpoints: Number of viewpoints to position
        - num_swarm_particles: Number of particles in the PSO swarm
        - grid_size: Size of the grid
        - w: Inertia weight
        - c1: Cognitive parameter (personal best influence)
        - c2: Social parameter (global best influence)
        """
        self.field = field
        self.num_viewpoints = num_viewpoints
        self.num_swarm_particles = num_swarm_particles
        self.grid_size = grid_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Initialize PSO swarm particles
        self.swarm = [PSOParticle(grid_size, num_viewpoints) for _ in range(num_swarm_particles)]
        
        # Initialize global best
        self.global_best_position = None
        self.global_best_fitness = -np.inf
        
        # History for visualization
        self.history = []
        
        # Initialize particles by evaluating them
        self._evaluate_all_particles()
        
    def _evaluate_fitness(self, position):
        """Compute fitness (coverage) for a given position configuration"""
        return self.field.compute_coverage_monte_carlo(position, num_samples=10000, threshold=0.25)
    
    def _evaluate_all_particles(self):
        """Evaluate all particles in the swarm"""
        for particle in self.swarm:
            particle.fitness = self._evaluate_fitness(particle.position)
            
            # Update personal best
            if particle.fitness > particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_position = particle.position.copy()
            
            # Update global best
            if particle.fitness > self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.position.copy()
    
    def step(self):
        """Perform one step of the PSO algorithm"""
        current_state = {
            'particles': [particle.position.copy() for particle in self.swarm],
            'fitnesses': [particle.fitness for particle in self.swarm],
            'personal_bests': [particle.best_position.copy() for particle in self.swarm],
            'personal_best_fitnesses': [particle.best_fitness for particle in self.swarm],
            'global_best_position': self.global_best_position.copy(),
            'global_best_fitness': self.global_best_fitness
        }
        
        for particle in self.swarm:
            # Update velocity using PSO equation
            r1, r2 = np.random.random(2)  # Random coefficients
            
            # Standard PSO velocity update
            inertia = self.w * particle.velocity
            cognitive = self.c1 * r1 * (particle.best_position - particle.position)
            social = self.c2 * r2 * (self.global_best_position - particle.position)
            
            particle.velocity = inertia + cognitive + social
            
            # Apply velocity limits to prevent explosion
            velocity_limit = 5.0
            particle.velocity = np.clip(particle.velocity, -velocity_limit, velocity_limit)
            
            # Update position
            particle.position = particle.position + particle.velocity
            
            # Apply boundary constraints with wrapping
            particle.position = particle.position % self.grid_size
            
            # Evaluate new position
            particle.fitness = self._evaluate_fitness(particle.position)
            
            # Update personal best
            if particle.fitness > particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_position = particle.position.copy()
            
            # Update global best
            if particle.fitness > self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.position.copy()
        
        # Record current state
        self.history.append(current_state)
        
        return self.global_best_position, self.global_best_fitness


# Initialize field for coverage calculations
field = Field(grid_size, fov_radius)

# Initialize PSO algorithm with tuned parameters
pso = PSO(field, num_viewpoints, num_swarm_particles, grid_size, 
          w=0.6,    # Lower inertia for faster convergence
          c1=1.2,   # Moderate cognitive component
          c2=2.0)   # Stronger social component to emphasize global best

# === FIGURE SETUP ===
fig = plt.figure(figsize=(18, 10), dpi=150, facecolor='white')
gs = GridSpec(3, 3, height_ratios=[3, 3, 1.2])

# Main coverage visualization plot
ax_main = fig.add_subplot(gs[0:2, 0:2])
ax_main.set_xlim(0, grid_size - 1)
ax_main.set_ylim(0, grid_size - 1)
ax_main.set_title("Coverage Optimization via PSO", fontsize=16, fontweight='bold')
ax_main.set_xlabel("X Coordinate", fontsize=12)
ax_main.set_ylabel("Y Coordinate", fontsize=12)
ax_main.set_aspect("equal")

# Force field visualization
ax_force = fig.add_subplot(gs[0:2, 2])
ax_force.set_xlim(0, grid_size - 1)
ax_force.set_ylim(0, grid_size - 1)
ax_force.set_title("Potential Field & Forces", fontsize=14)
ax_force.set_xlabel("X Coordinate", fontsize=12)
ax_force.set_ylabel("Y Coordinate", fontsize=12)
ax_force.set_aspect("equal")

# Coverage plot
ax_coverage = fig.add_subplot(gs[2, :])
ax_coverage.set_xlim(0, frames)
ax_coverage.set_ylim(0, 1)
ax_coverage.set_xlabel("Iteration", fontsize=12)
ax_coverage.set_ylabel("Coverage", fontsize=12)
ax_coverage.set_title("Coverage Optimization Progress", fontsize=14)
ax_coverage.grid(True, linestyle='--', alpha=0.7)

# Initialize data structures
coverage_data = []
velocity_norms = []
current_viewpoints = pso.global_best_position.copy()

# Update visibility for initial state
field.update_visibility(current_viewpoints)

# Create coverage plot
coverage_line, = ax_coverage.plot([], [], color='#0052cc', linewidth=3)

# Velocity norm plot
velocity_line, = ax_coverage.plot([], [], color='#dd0000', linewidth=2, alpha=0.7, linestyle='--')

# Create legend
ax_coverage.legend(['Coverage', 'Avg. Velocity'], loc='lower right')

# Create visibility plot with custom colormap
vis_plot = ax_main.imshow(
    field.visibility,
    extent=[0, grid_size, 0, grid_size],
    origin='lower',
    cmap=blue_cream,
    vmin=0,
    vmax=1,
    interpolation='nearest'
)

# Add colorbar for visibility
divider = make_axes_locatable(ax_main)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(vis_plot, cax=cax)
cbar.set_label("Coverage", fontsize=12)

# Viewpoint scatter
scatter = ax_main.scatter(
    [], [], 
    c='#fc4768',
    s=100, 
    zorder=5,
    edgecolor='white',
    linewidth=1.5
)

# Initialize force field visualization
potential_plot = ax_force.imshow(
    np.zeros((grid_size, grid_size)),
    extent=[0, grid_size, 0, grid_size],
    origin='lower',
    cmap=potential_cmap,
    interpolation='nearest'
)

# Add colorbar for potential
divider = make_axes_locatable(ax_force)
cax_force = divider.append_axes("right", size="5%", pad=0.1)
cbar_force = plt.colorbar(potential_plot, cax=cax_force)
cbar_force.set_label("Potential", fontsize=12)

# Quiver plot for force field
quiver = None

# FOV circles for viewpoints
circles = []
for i in range(num_viewpoints):
    circle = Circle(
        (0, 0),  # Will be updated
        fov_radius,
        fill=False,
        edgecolor='#fc4768',
        alpha=0.6,
        linestyle='-',
        linewidth=1.5
    )
    ax_main.add_patch(circle)
    circles.append(circle)

# Particle exploration markers
exploration_scatter = ax_main.scatter(
    [], [], 
    c='#7a76fc',
    s=50, 
    alpha=0.4,
    marker='x'
)

# Text elements
coverage_text = None
iteration_text = None
best_solution_text = None

plt.tight_layout()

def update(frame):
    global coverage_text, iteration_text, best_solution_text, quiver
    
    # Step the PSO algorithm if not the first frame
    if frame > 0:
        best_position, best_fitness = pso.step()
    
    # Get current state from history
    if frame < len(pso.history) or frame == 0:
        if frame == 0:
            state = {
                'global_best_position': pso.global_best_position,
                'global_best_fitness': pso.global_best_fitness,
                'particles': [p.position for p in pso.swarm],
                'fitnesses': [p.fitness for p in pso.swarm]
            }
        else:
            state = pso.history[frame-1]
        
        # Get global best
        global_best_position = state['global_best_position']
        global_best_fitness = state['global_best_fitness']
        
        # Compute average velocity norm for visualization
        if frame > 0:
            velocities = [p.velocity for p in pso.swarm]
            avg_velocity = np.mean([np.linalg.norm(v) for v in velocities])
            velocity_norms.append(avg_velocity)
        else:
            velocity_norms.append(0)
        
        # Update coverage
        coverage = global_best_fitness
        coverage_data.append(coverage)
        
        # Update visibility map for best solution
        field.update_visibility(global_best_position)
        vis_plot.set_data(field.visibility)
        
        # Update viewpoint positions
        scatter.set_offsets(global_best_position)
        
        # Update FOV circles
        for i, circle in enumerate(circles):
            circle.center = (global_best_position[i, 0], global_best_position[i, 1])
        
        # Show explorations from all particles
        exploration_points = np.vstack(state['particles'])
        exploration_scatter.set_offsets(exploration_points)
        
        # Update coverage and velocity lines
        coverage_line.set_data(range(len(coverage_data)), coverage_data)
        velocity_line.set_data(range(len(velocity_norms)), np.array(velocity_norms) / 10)  # Scale for visibility
        
        # Compute potential field and forces for visualization
        field.compute_potential(global_best_position)
        force_field, SX, SY = field.compute_forces(global_best_position)
        
        # Update potential field plot
        potential_plot.set_data(field.potential)
        potential_plot.set_clim(vmin=0, vmax=np.max(field.potential) * 1.2)
        
        # Update force field quiver plot
        if quiver:
            quiver.remove()
        
        # Scale forces for better visualization
        force_magnitudes = np.linalg.norm(force_field, axis=2)
        max_mag = np.max(force_magnitudes) if np.max(force_magnitudes) > 0 else 1
        normalized_forces = force_field / max_mag
        
        quiver = ax_force.quiver(
            SX, SY,
            normalized_forces[:, :, 0], normalized_forces[:, :, 1],
            force_magnitudes,
            cmap='hot',
            scale=25,
            headwidth=4,
            headlength=5,
            headaxislength=4.5,
            width=0.003
        )
        
        # Remove previous text (if exists)
        if coverage_text:
            coverage_text.remove()
        if iteration_text:
            iteration_text.remove()
        if best_solution_text:
            best_solution_text.remove()
        
        # Add new text with stylish formatting
        coverage_text = ax_coverage.text(
            0.98, 0.15, 
            f"Coverage: {coverage*100:.2f}%", 
            ha='right', va='bottom',
            transform=ax_coverage.transAxes, 
            fontsize=12, 
            color='#0052cc',
            bbox=dict(facecolor='white', edgecolor='#0052cc', alpha=0.8, 
                      boxstyle='round,pad=0.5', linewidth=1.5)
        )
        
        iteration_text = ax_main.text(
            0.03, 0.97, 
            f"Iteration: {frame}",
            ha='left', va='top',
            transform=ax_main.transAxes,
            fontsize=12,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3')
        )
        
        best_solution_text = ax_force.text(
            0.5, 0.03, 
            f"Viewpoints: {num_viewpoints}",
            ha='center', va='bottom',
            transform=ax_force.transAxes,
            fontsize=12,
            color='#333333',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3')
        )
        
        # Dynamic x-axis adjustment for coverage plot
        ax_coverage.set_xlim(0, max(len(coverage_data) + 5, 50))
        
        # Print progress
        print(f"Iteration {frame}: Best Coverage = {global_best_fitness*100:.2f}%, " +
              f"Avg Velocity = {velocity_norms[-1]:.3f}")
    
    return (vis_plot, potential_plot, scatter, exploration_scatter, coverage_line, velocity_line, 
            quiver, coverage_text, iteration_text, best_solution_text, *circles)

# Create animation with higher quality
ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=False)

plt.tight_layout()
plt.show()

# Save animation with high quality
ani.save('pso_coverage_optimization_refined.mp4', writer='ffmpeg', fps=24, dpi=300)