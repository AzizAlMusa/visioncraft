import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

# CONFIGURATION OPTIONS
USE_GAUSSIAN_REPULSION = True  # Set to False for hard inverse-distance repulsion
SHOW_PLANNED_TARGET = True  # Show the planned target with X marker
PARTICLE_RING_RADIUS = 2.5  # Radius of influence ring around particle
OBSTACLE_RING_MULTIPLIER = 1.5  # Multiplier for obstacle ring radius (obstacle_radius * this)
SAVE_ANIMATION = True  # Set to True to save as MP4
ANIMATION_FILENAME = "potential_field_navigation_1.mp4"  # Output filename

# Set up the figure with dark theme - two subplots SIDE BY SIDE
plt.style.use('dark_background')
fig = plt.figure(figsize=(20, 10))  # Wider figure for side-by-side layout

# 3D plot on left
ax3d = fig.add_subplot(121, projection='3d')
# 2D plot on right
ax2d = fig.add_subplot(122)

# Grid parameters - expanded space
x_min, x_max = -8, 8
y_min, y_max = -8, 8
resolution = 120
x = np.linspace(x_min, x_max, resolution)
y = np.linspace(y_min, y_max, resolution)
X, Y = np.meshgrid(x, y)

# Target location
target = np.array([3.5, -3.5])

# Define obstacles (centers and radii)
obstacles = [
    {'center': np.array([1.5, 2.0]), 'radius': 2.0, 'strength': 15},
    {'center': np.array([-4.0, 1.0]), 'radius': 2.0, 'strength': 15},
    {'center': np.array([3.5, -3.5]), 'radius': 2.0, 'strength': 15}
]

def compute_potential_field(X, Y, obstacles, target, use_gaussian=True, particle_pos=None, particle_ring_radius=1.0):
    """Compute the potential field with attractive and repulsive forces"""
    # Attractive potential (quadratic bowl toward target)
    attractive = 0.5 * ((X - target[0])**2 + (Y - target[1])**2)
    
    # Repulsive potential from obstacles
    repulsive = np.zeros_like(X)
    for obs in obstacles:
        # Distance from obstacle center to grid points
        dist_to_obstacle_center = np.sqrt((X - obs['center'][0])**2 + (Y - obs['center'][1])**2)
        
        # If particle position is provided, check if particle ring intersects with obstacle's influence ring
        if particle_pos is not None:
            # Distance from particle center to obstacle center
            particle_to_obstacle_dist = np.linalg.norm(particle_pos - obs['center'])
            obstacle_ring_radius = obs['radius'] * OBSTACLE_RING_MULTIPLIER
            
            # Calculate the minimum distance between particle ring edge and obstacle ring edge
            # This is the distance between centers minus both ring radii
            ring_edge_to_ring_edge_dist = particle_to_obstacle_dist - particle_ring_radius - obstacle_ring_radius
            
            # Only apply repulsion if the rings are touching or overlapping (distance <= 0)
            if ring_edge_to_ring_edge_dist > 0:
                continue
        
        if use_gaussian:
            # Soft Gaussian repulsive field - smooth bumps
            repulsive += obs['strength'] * np.exp(-0.5 * (dist_to_obstacle_center / obs['radius'])**2)
        else:
            # Hard repulsive field: 1/d with safety distance - cliff-like barriers
            # For ring-based calculation, we need to consider the distance from the particle ring edge
            if particle_pos is not None:
                # Calculate distance from each grid point to the particle center
                dist_to_particle_center = np.sqrt((X - particle_pos[0])**2 + (Y - particle_pos[1])**2)
                
                # For points inside the particle ring, use the ring radius as the effective distance
                # For points outside, use the actual distance minus the ring radius
                effective_particle_dist = np.maximum(dist_to_particle_center - particle_ring_radius, 0)
                
                # Distance from particle ring edge to obstacle center
                dist_from_particle_ring_to_obstacle = np.sqrt(
                    (X - particle_pos[0])**2 + (Y - particle_pos[1])**2
                ) - particle_ring_radius
                
                # But we want the repulsion to be based on distance from obstacle, so use original calculation
                # but only where rings are intersecting (which we already checked above)
                safety_dist = 0.1  # Prevent division by zero
                effective_dist = np.maximum(dist_to_obstacle_center - obs['radius'], safety_dist)
                repulsive += obs['strength'] / effective_dist
            else:
                # Standard calculation when no particle position is given
                safety_dist = 0.1
                effective_dist = np.maximum(dist_to_obstacle_center - obs['radius'], safety_dist)
                repulsive += obs['strength'] / effective_dist
    
    return attractive + repulsive

def draw_ring_3d(ax, center_x, center_y, center_z, radius, color='white', alpha=0.6):
    """Draw a 3D ring (circle) at given position"""
    theta = np.linspace(0, 2*np.pi, 50)
    ring_x = center_x + radius * np.cos(theta)
    ring_y = center_y + radius * np.sin(theta)
    ring_z = np.full_like(ring_x, center_z + 0.2)  # Slightly above surface
    ax.plot(ring_x, ring_y, ring_z, color=color, linewidth=2, alpha=alpha)

def draw_ring_2d(ax, center_x, center_y, radius, color='white', alpha=0.6, linewidth=2):
    """Draw a 2D ring (circle) at given position"""
    theta = np.linspace(0, 2*np.pi, 50)
    ring_x = center_x + radius * np.cos(theta)
    ring_y = center_y + radius * np.sin(theta)
    ax.plot(ring_x, ring_y, color=color, linewidth=linewidth, alpha=alpha)

def compute_gradient(X, Y, potential):
    """Compute gradient of potential field"""
    grad_y, grad_x = np.gradient(potential)
    return grad_x, grad_y

def rings_are_intersecting(particle_pos, particle_radius, obstacle_center, obstacle_radius):
    """Check if particle ring intersects with obstacle ring"""
    center_distance = np.linalg.norm(particle_pos - obstacle_center)
    return center_distance <= (particle_radius + obstacle_radius)

# Compute potential field (initial computation without particle position)
Z = compute_potential_field(X, Y, obstacles, target, USE_GAUSSIAN_REPULSION)

# Compute gradient for movement direction
grad_x, grad_y = compute_gradient(X, Y, Z)

# Starting position - further out to show more movement
start_pos = np.array([-6.5, 6.5])
current_pos = start_pos.copy()

# Adam optimizer parameters for momentum-based movement
beta1 = 0.9  # Exponential decay rate for first moment estimates
beta2 = 0.999  # Exponential decay rate for second moment estimates
epsilon = 1e-8  # Small constant for numerical stability
learning_rate = 0.15  # Step size

# Initialize Adam optimizer state
m = np.zeros(2)  # First moment vector (momentum)
v = np.zeros(2)  # Second moment vector (adaptive learning rates)
t = 0  # Time step counter

# Path tracking - allow more steps for longer journey
path = [current_pos.copy()]
max_steps = 500

# Track which obstacles are currently affecting the particle
active_obstacles = []

# Simulate path using Adam optimizer for smooth movement
for step in range(max_steps):
    # Get current position indices
    i = int((current_pos[0] - x_min) / (x_max - x_min) * (resolution - 1))
    j = int((current_pos[1] - y_min) / (y_max - y_min) * (resolution - 1))
    
    # Clamp indices
    i = max(0, min(resolution - 1, i))
    j = max(0, min(resolution - 1, j))
    
    # Check which obstacles are currently intersecting with particle ring
    current_active_obstacles = []
    for obs_idx, obs in enumerate(obstacles):
        obstacle_ring_radius = obs['radius'] * OBSTACLE_RING_MULTIPLIER
        if rings_are_intersecting(current_pos, PARTICLE_RING_RADIUS, obs['center'], obstacle_ring_radius):
            current_active_obstacles.append(obs_idx)
    
    active_obstacles.append(current_active_obstacles.copy())
    
    # Get gradient at current position (recompute potential field considering particle position)
    local_Z = compute_potential_field(X, Y, obstacles, target, USE_GAUSSIAN_REPULSION, current_pos, PARTICLE_RING_RADIUS)
    local_grad_x, local_grad_y = compute_gradient(X, Y, local_Z)
    gradient = np.array([-local_grad_x[j, i], -local_grad_y[j, i]])  # Negative for gradient descent
    
    # Adam optimizer update
    t += 1  # Increment time step
    
    # Update biased first moment estimate (momentum)
    m = beta1 * m + (1 - beta1) * gradient
    
    # Update biased second moment estimate (adaptive learning rate)
    v = beta2 * v + (1 - beta2) * (gradient ** 2)
    
    # Compute bias-corrected first moment estimate
    m_hat = m / (1 - beta1 ** t)
    
    # Compute bias-corrected second moment estimate
    v_hat = v / (1 - beta2 ** t)
    
    # Update position using Adam
    update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    current_pos += update
    
    path.append(current_pos.copy())
    
    # Stop if close to target
    if np.linalg.norm(current_pos - target) < 0.3:
        break

path = np.array(path)

# Create colormap for surface
colors = plt.cm.plasma(np.linspace(0, 1, 256))
colors[0] = [0.1, 0.1, 0.3, 0.8]  # Dark base color
colormap = mcolors.LinearSegmentedColormap.from_list('custom', colors)

# === 3D PLOT SETUP ===
# Plot surface with transparency
surface = ax3d.plot_surface(X, Y, Z, cmap=colormap, alpha=0.7, 
                           linewidth=0, antialiased=True, shade=True)

# Find the actual minimum in the potential field near the target
target_region_mask = (np.abs(X - target[0]) < 2.0) & (np.abs(Y - target[1]) < 2.0)
if np.any(target_region_mask):
    min_idx = np.unravel_index(np.argmin(Z[target_region_mask]), Z[target_region_mask].shape)
    target_region_indices = np.where(target_region_mask)
    actual_min_i = target_region_indices[0][min_idx[0]]
    actual_min_j = target_region_indices[1][min_idx[0]]
    actual_target_x = X[actual_min_i, actual_min_j]
    actual_target_y = Y[actual_min_i, actual_min_j]
    actual_target_z = Z[actual_min_i, actual_min_j]
else:
    actual_target_x, actual_target_y = target[0], target[1]
    actual_target_z = compute_potential_field(target[0], target[1], obstacles, target, USE_GAUSSIAN_REPULSION)

if SHOW_PLANNED_TARGET:
    # Mark the planned target with a nice red X in 3D
    x_size = 0.3
    planned_target_z = compute_potential_field(target[0], target[1], obstacles, target, USE_GAUSSIAN_REPULSION)
    
    # First diagonal of X
    x_line1_x = [target[0] - x_size, target[0] + x_size]
    x_line1_y = [target[1] - x_size, target[1] + x_size]
    x_line1_z = [planned_target_z + 0.5, planned_target_z + 0.5]
    # Second diagonal of X
    x_line2_x = [target[0] - x_size, target[0] + x_size]
    x_line2_y = [target[1] + x_size, target[1] - x_size]
    x_line2_z = [planned_target_z + 0.5, planned_target_z + 0.5]
    
    ax3d.plot(x_line1_x, x_line1_y, x_line1_z, color='#ff1744', linewidth=4, alpha=1.0)
    ax3d.plot(x_line2_x, x_line2_y, x_line2_z, color='#ff1744', linewidth=4, alpha=1.0)
    # Add glow effect
    ax3d.plot(x_line1_x, x_line1_y, x_line1_z, color='#ff1744', linewidth=8, alpha=0.3)
    ax3d.plot(x_line2_x, x_line2_y, x_line2_z, color='#ff1744', linewidth=8, alpha=0.3)

# Mark obstacles with high-contrast colors and draw their influence rings in 3D
for obs in obstacles:
    obs_z = compute_potential_field(obs['center'][0], obs['center'][1], obstacles, target, USE_GAUSSIAN_REPULSION)
    ax3d.scatter(obs['center'][0], obs['center'][1], obs_z, 
                color='#ff1744', s=150, alpha=0.9)
    ax3d.scatter(obs['center'][0], obs['center'][1], obs_z, 
                color='#ff1744', s=300, alpha=0.4)
    
    # Draw white ring around obstacle showing influence zone
    ring_radius = obs['radius'] * OBSTACLE_RING_MULTIPLIER
    draw_ring_3d(ax3d, obs['center'][0], obs['center'][1], obs_z, ring_radius, 'white', 0.8)

# === 2D PLOT SETUP ===
# Plot potential field as contours
contour = ax2d.contourf(X, Y, Z, levels=50, cmap=colormap, alpha=0.8)
contour_lines = ax2d.contour(X, Y, Z, levels=20, colors='white', alpha=0.3, linewidths=0.5)

# Mark obstacles in 2D
for obs in obstacles:
    # Draw obstacle center
    ax2d.scatter(obs['center'][0], obs['center'][1], 
                color='#ff1744', s=150, alpha=0.9, zorder=10)
    ax2d.scatter(obs['center'][0], obs['center'][1], 
                color='#ff1744', s=300, alpha=0.4, zorder=9)
    
    # Draw obstacle boundary circle
    obstacle_circle = plt.Circle(obs['center'], obs['radius'], 
                               fill=False, color='#ff1744', linewidth=2, alpha=0.8)
    ax2d.add_patch(obstacle_circle)
    
    # Draw white ring around obstacle showing influence zone
    ring_radius = obs['radius'] * OBSTACLE_RING_MULTIPLIER
    draw_ring_2d(ax2d, obs['center'][0], obs['center'][1], ring_radius, 'white', 0.8, 2)

if SHOW_PLANNED_TARGET:
    # Mark the planned target with a nice red X in 2D
    x_size = 0.3
    ax2d.plot([target[0] - x_size, target[0] + x_size], 
              [target[1] - x_size, target[1] + x_size], 
              color='#ff1744', linewidth=4, alpha=1.0, zorder=10)
    ax2d.plot([target[0] - x_size, target[0] + x_size], 
              [target[1] + x_size, target[1] - x_size], 
              color='#ff1744', linewidth=4, alpha=1.0, zorder=10)
    # Add glow effect
    ax2d.plot([target[0] - x_size, target[0] + x_size], 
              [target[1] - x_size, target[1] + x_size], 
              color='#ff1744', linewidth=8, alpha=0.3, zorder=9)
    ax2d.plot([target[0] - x_size, target[0] + x_size], 
              [target[1] + x_size, target[1] - x_size], 
              color='#ff1744', linewidth=8, alpha=0.3, zorder=9)

# Initialize animated elements for both plots
# 3D elements
particle_3d = ax3d.scatter([], [], [], color='#ffff00', s=100, alpha=1.0)
trail_line_3d, = ax3d.plot([], [], [], color='#ffff00', linewidth=2, alpha=0.8)
particle_ring_lines_3d = []

# 2D elements
particle_2d = ax2d.scatter([], [], color='#ffff00', s=100, alpha=1.0, zorder=15)
trail_line_2d, = ax2d.plot([], [], color='#ffff00', linewidth=2, alpha=0.8, zorder=12)
particle_ring_2d = None  # Will be created in animation

# Set up the 3D plot
ax3d.set_xlim(x_min, x_max)
ax3d.set_ylim(y_min, y_max)
ax3d.set_zlim(Z.min() - 1, Z.max() + 2)
ax3d.set_axis_off()
ax3d.grid(False)
ax3d.set_box_aspect([1,1,0.8])
ax3d.view_init(elev=25, azim=45)
ax3d.set_title('3D View - Ring-Based Potential Field Navigation', color='white', fontsize=14, pad=20)

# Set up the 2D plot
ax2d.set_xlim(x_min, x_max)
ax2d.set_ylim(y_min, y_max)
ax2d.set_aspect('equal')
ax2d.set_xlabel('X', color='white', fontsize=12)
ax2d.set_ylabel('Y', color='white', fontsize=12)
ax2d.set_title('Top View - Ring-Based Potential Field Navigation', color='white', fontsize=14)
ax2d.tick_params(colors='white')

# Animation function
def animate(frame):
    global particle_ring_lines_3d, particle_ring_2d
    
    if frame >= len(path):
        return [particle_3d, trail_line_3d, particle_2d, trail_line_2d] + particle_ring_lines_3d + ([particle_ring_2d] if particle_ring_2d else [])
    
    # Current position
    pos = path[frame]
    pos_z = compute_potential_field(pos[0], pos[1], obstacles, target, USE_GAUSSIAN_REPULSION)
    
    # Update 3D particle
    particle_3d._offsets3d = ([pos[0]], [pos[1]], [pos_z])
    
    # Update 2D particle
    particle_2d.set_offsets([[pos[0], pos[1]]])
    
    # Remove old 3D particle ring
    for ring_line in particle_ring_lines_3d:
        ring_line.remove()
    particle_ring_lines_3d = []
    
    # Remove old 2D particle ring
    if particle_ring_2d:
        particle_ring_2d.remove()
        particle_ring_2d = None
    
    # Determine ring color based on whether it's intersecting with any obstacle rings
    ring_color = 'white'
    ring_alpha = 0.6
    if frame < len(active_obstacles) and active_obstacles[frame]:
        ring_color = 'orange'  # Orange when intersecting with obstacle rings
        ring_alpha = 0.9
    
    # Draw ring around particle in 3D
    theta = np.linspace(0, 2*np.pi, 50)
    ring_x = pos[0] + PARTICLE_RING_RADIUS * np.cos(theta)
    ring_y = pos[1] + PARTICLE_RING_RADIUS * np.sin(theta)
    ring_z = np.full_like(ring_x, pos_z + 0.3)  # Slightly above particle
    ring_line_3d, = ax3d.plot(ring_x, ring_y, ring_z, color=ring_color, linewidth=2, alpha=ring_alpha)
    particle_ring_lines_3d.append(ring_line_3d)
    
    # Draw ring around particle in 2D
    particle_ring_2d = plt.Circle(pos, PARTICLE_RING_RADIUS, 
                                 fill=False, color=ring_color, linewidth=2, alpha=ring_alpha)
    ax2d.add_patch(particle_ring_2d)
    
    # Update permanent trails - show all path up to current frame
    if frame > 0:
        trail_positions = path[:frame+1]
        
        # 3D trail
        trail_z = [compute_potential_field(p[0], p[1], obstacles, target, USE_GAUSSIAN_REPULSION) 
                   for p in trail_positions]
        trail_line_3d.set_data_3d(trail_positions[:, 0], 
                                  trail_positions[:, 1], 
                                  trail_z)
        
        # 2D trail
        trail_line_2d.set_data(trail_positions[:, 0], trail_positions[:, 1])
    
    return [particle_3d, trail_line_3d, particle_2d, trail_line_2d] + particle_ring_lines_3d + ([particle_ring_2d] if particle_ring_2d else [])

# Create animation
anim = FuncAnimation(fig, animate, frames=len(path)+50, 
                    interval=50, blit=False, repeat=True)

# Save animation as MP4 if requested
if SAVE_ANIMATION:
    print(f"Saving animation as {ANIMATION_FILENAME}...")
    print("This may take a few minutes for HD quality...")
    
    # Use ffmpeg writer with high quality settings for HD
    Writer = matplotlib.animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='Potential Field Simulation'), bitrate=5000)
    
    # Save with HD quality (1920x1080 equivalent DPI)
    anim.save(ANIMATION_FILENAME, writer=writer, dpi=150, savefig_kwargs={'facecolor':'black'})
    print(f"Animation saved successfully as {ANIMATION_FILENAME}")

plt.tight_layout()
plt.show()

# Print configuration and simulation info
repulsion_type = "Gaussian (soft bumps)" if USE_GAUSSIAN_REPULSION else "Hard inverse-distance (cliff-like)"

print(f"Path length: {len(path)} steps")
print(f"Final distance to target: {np.linalg.norm(path[-1] - target):.3f}")
print(f"Repulsion type: {repulsion_type}")
print(f"Particle ring radius: {PARTICLE_RING_RADIUS}")
print("\nRing-Based Repulsion System:")
print("• Repulsion is calculated from the EDGE of the particle's ring, not its center")
print("• Obstacles only affect the particle when their influence rings TOUCH the particle ring")
print("• Particle ring turns ORANGE when intersecting with obstacle influence zones")
print("• This creates more realistic collision avoidance behavior")
print("\nSide-by-Side Visualization shows:")
print("• LEFT: 3D perspective with height representing potential field")
print("• RIGHT: Top-down view with contour lines showing potential field")
print("• Yellow particle using Adam optimizer with momentum")
print(f"• Red obstacles with {repulsion_type.lower()}")
print("• Red X marking the planned target location")
print("• White rings around obstacles showing influence zones")
print("• White/Orange ring around particle (Orange = active repulsion)")
print("• Permanent golden trail showing smooth momentum-based path")
print("• Plasma colormap showing potential field topology")
print("\nTo switch repulsion type, change USE_GAUSSIAN_REPULSION at the top of the code")
print("To toggle target marker, change SHOW_PLANNED_TARGET at the top of the code")
if SAVE_ANIMATION:
    print(f"\nAnimation saved as HD MP4: {ANIMATION_FILENAME}")
    print("• 20 FPS for smooth playback")
    print("• High bitrate (5000) for quality")
    print("• DPI 150 for HD resolution")
else:
    print("\nTo save as MP4, set SAVE_ANIMATION = True at the top of the code")