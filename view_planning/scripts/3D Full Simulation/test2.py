import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.font_manager as fm

# Set up custom font - IBM Plex Mono
try:
    font_path = './font/IBMPlexMono-Regular.ttf'
    custom_font = fm.FontProperties(fname=font_path)
    bold_font = fm.FontProperties(fname='./font/IBMPlexMono-Bold.ttf')
    print(f"Custom font loaded: {font_path}")
except:
    # Fallback to system fonts
    custom_font = fm.FontProperties(family='monospace')
    bold_font = fm.FontProperties(family='monospace', weight='bold')
    print("Using fallback monospace font")

# Set up the retro aesthetic with HD figure size
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 12), dpi=150)  # Increased size and DPI for HD
fig.patch.set_facecolor('#0a0a0a')
ax = fig.add_subplot(111, projection='3d')

# Create sphere wireframe with thicker lines
u = np.linspace(0, 2 * np.pi, 25)
v = np.linspace(0, np.pi, 20)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the sphere wireframe with thicker lines
sphere_wireframe = ax.plot_wireframe(x_sphere, y_sphere, z_sphere, 
                                   color='#00ff41', alpha=0.6, linewidth=1.2)

# Set up the retro styling - hide panes completely
ax.set_facecolor('#0a0a0a')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('#0a0a0a')  # Same as background - invisible
ax.yaxis.pane.set_edgecolor('#0a0a0a')
ax.zaxis.pane.set_edgecolor('#0a0a0a')
ax.xaxis.pane.set_alpha(0)  # Completely transparent
ax.yaxis.pane.set_alpha(0)
ax.zaxis.pane.set_alpha(0)

# Hide the axes spines/lines
ax.xaxis.line.set_color('#0a0a0a')
ax.yaxis.line.set_color('#0a0a0a')
ax.zaxis.line.set_color('#0a0a0a')
ax.xaxis.set_pane_color((0, 0, 0, 0))
ax.yaxis.set_pane_color((0, 0, 0, 0))
ax.zaxis.set_pane_color((0, 0, 0, 0))

# Customize axes with retro font - but turn off visibility
ax.set_xlabel('', color='#00ff41', fontproperties=custom_font, fontweight='bold')
ax.set_ylabel('', color='#00ff41', fontproperties=custom_font, fontweight='bold')
ax.set_zlabel('', color='#00ff41', fontproperties=custom_font, fontweight='bold')

# Hide all axes elements
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.tick_params(colors='#00ff41', labelsize=8, length=0)  # Zero length ticks

# Set axis limits and equal aspect ratio
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])

# Force equal aspect ratio for all axes
ax.set_box_aspect([1,1,1])  # Equal aspect ratio

# Set initial viewing angle
initial_elev = 20
initial_azim = 30
rotation_degrees = 75  # Total rotation in degrees
ax.view_init(elev=initial_elev, azim=initial_azim)

# Add retro title
ax.set_title('POTENTIAL FIELD ATTRACTION-REPULSION SIMULATION', 
             color='#00ff41', fontsize=18, fontweight='bold', 
             pad=30, fontproperties=bold_font)

# Physics parameters
attraction_strength = 1.2  # Slightly increased for faster motion
repulsion_strength = 20.0  # Massively increased - much stronger than original
damping = 0.95  # Reduced damping for more energetic motion
dt = 0.03  # Increased time step for faster animation
gaussian_sigma = 0.30  # Doubled from 0.15 - much larger bumps!
gaussian_range = 3 * gaussian_sigma  # Effective range (3-sigma rule) - now 0.9 units

# Animation parameters for 10 seconds
fps = 30  # Frames per second
duration = 10  # Duration in seconds
total_frames = fps * duration  # Total frames = 300
interval = 1000 / fps  # Interval in milliseconds = 33.33ms

print(f"Creating {total_frames} frames at {fps} FPS for {duration} seconds")

# Initialize positions and velocities for two dots
# Dot 1: Start at north pole
pos1 = np.array([0.0, 0.0, 1.0])
vel1 = np.array([0.1, 0.0, 0.0])

# Dot 2: Start at south pole  
pos2 = np.array([0.0, 0.0, -1.0])
vel2 = np.array([-0.1, 0.0, 0.0])

# Attraction point (fixed red dot)
attractor_pos = np.array([0.8, 0.6, 0.2])
attractor_pos = attractor_pos / np.linalg.norm(attractor_pos)  # Normalize to sphere

# Initialize trail lists
trail1_x, trail1_y, trail1_z = [], [], []
trail2_x, trail2_y, trail2_z = [], [], []

# Initialize plot elements
dot1, = ax.plot([], [], [], 'o', color='#ffff00', markersize=12)
dot2, = ax.plot([], [], [], 'o', color='#00ffff', markersize=12)

trail_line1, = ax.plot([], [], [], color='#ffff00', linewidth=2.5, alpha=0.8)
trail_points1, = ax.plot([], [], [], 'o', color='#ffff00', markersize=2.5, alpha=0.6)

trail_line2, = ax.plot([], [], [], color='#00ffff', linewidth=2.5, alpha=0.8)
trail_points2, = ax.plot([], [], [], 'o', color='#00ffff', markersize=2.5, alpha=0.6)

# Initialize Gaussian bumps and attractor arrows
bump_surface1 = None
bump_surface2 = None
attractor_arrows = []

# Add some retro text annotations with larger font sizes
fig.text(0.02, 0.95, 'MULTI-AGENT PHYSICS SIMULATION', 
         color='#00ff41', fontsize=14, fontproperties=bold_font)
fig.text(0.02, 0.92, 'STATUS: ACTIVE', 
         color='#00ff41', fontsize=12, fontproperties=custom_font)
fig.text(0.02, 0.89, 'MODE: ATTRACTION-REPULSION DYNAMICS', 
         color='#00ff41', fontsize=12, fontproperties=custom_font)

def project_to_sphere(pos):
    """Project position to unit sphere surface"""
    return pos / np.linalg.norm(pos)

def tangent_component(vec, normal):
    """Get tangent component of vector (remove normal component)"""
    return vec - np.dot(vec, normal) * normal

# Display attractor as arrow field pointing toward it
def create_attractor_arrows():
    """Create arrows pointing toward the attractor position"""
    # Clear previous arrows
    for arrow in attractor_arrows:
        arrow.remove()
    attractor_arrows.clear()
    
    # Create a ring of arrows around the attractor
    n_arrows = 8
    arrow_distance = 0.3
    
    for i in range(n_arrows):
        angle = 2 * np.pi * i / n_arrows
        
        # Create arrow position around the attractor
        # Find two perpendicular vectors to attractor normal
        normal = attractor_pos
        if abs(normal[2]) < 0.9:
            v1 = np.cross(normal, [0, 0, 1])
        else:
            v1 = np.cross(normal, [1, 0, 0])
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Arrow start position
        arrow_offset = arrow_distance * (np.cos(angle) * v1 + np.sin(angle) * v2)
        arrow_start = attractor_pos + arrow_offset
        arrow_start = arrow_start / np.linalg.norm(arrow_start)  # Project to sphere
        
        # Arrow direction (toward attractor)
        arrow_direction = attractor_pos - arrow_start
        arrow_direction = arrow_direction / np.linalg.norm(arrow_direction) * 0.15
        
        # Create arrow
        arrow = ax.quiver(arrow_start[0], arrow_start[1], arrow_start[2],
                         arrow_direction[0], arrow_direction[1], arrow_direction[2],
                         color='#ff0000', alpha=0.8, arrow_length_ratio=0.3,
                         linewidth=2)
        attractor_arrows.append(arrow)

def create_gaussian_bump(center_pos, bump_height=0.25, sigma=None):
    """Create Gaussian bump at given position on sphere"""
    if sigma is None:
        sigma = gaussian_sigma
        
    normal = center_pos  # For unit sphere, normal = position
    
    # Create local coordinate system
    if abs(normal[2]) < 0.9:
        v1 = np.cross(normal, [0, 0, 1])
    else:
        v1 = np.cross(normal, [1, 0, 0])
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Create grid for the bump (size based on sigma)
    bump_size = 3 * sigma  # 3-sigma range
    u_bump = np.linspace(-bump_size, bump_size, 12)
    v_bump = np.linspace(-bump_size, bump_size, 12)
    U_bump, V_bump = np.meshgrid(u_bump, v_bump)
    
    # Calculate Gaussian height
    r_squared = U_bump**2 + V_bump**2
    gaussian = bump_height * np.exp(-r_squared / (2 * sigma**2))
    
    # Convert to 3D coordinates
    bump_x = np.zeros_like(U_bump)
    bump_y = np.zeros_like(U_bump)
    bump_z = np.zeros_like(U_bump)
    
    for i in range(U_bump.shape[0]):
        for j in range(U_bump.shape[1]):
            local_pos = U_bump[i,j] * v1 + V_bump[i,j] * v2
            sphere_pos = center_pos + local_pos
            sphere_pos = sphere_pos / np.linalg.norm(sphere_pos)
            final_pos = sphere_pos + gaussian[i,j] * normal
            bump_x[i,j] = final_pos[0]
            bump_y[i,j] = final_pos[1]
            bump_z[i,j] = final_pos[2]
    
    return bump_x, bump_y, bump_z

def gaussian_repulsion_force(pos1, pos2, strength, sigma):
    """Calculate repulsion force based on Gaussian overlap"""
    # Calculate distance between positions
    separation = pos2 - pos1
    distance = np.linalg.norm(separation)
    
    # Gaussian repulsion - strongest when distance is small
    # Force magnitude based on Gaussian function
    force_magnitude = strength * np.exp(-distance**2 / (2 * sigma**2))
    
    if distance > 0.001:  # Avoid division by zero
        force_direction = -separation / distance  # Repulsive force on pos1
        return force_magnitude * force_direction
    else:
        return np.array([0.0, 0.0, 0.0])

# Create initial attractor visualization
create_attractor_arrows()

# Animation function
def animate(frame):
    global pos1, pos2, vel1, vel2, trail1_x, trail1_y, trail1_z, trail2_x, trail2_y, trail2_z
    global bump_surface1, bump_surface2
    
    # Progress indicator
    if frame % 30 == 0:  # Every second
        print(f"Rendering frame {frame}/{total_frames} ({frame/total_frames*100:.1f}%)")
    
    # Smooth camera rotation over the animation duration
    rotation_progress = frame / total_frames  # 0 to 1
    current_azim = initial_azim + (rotation_degrees * rotation_progress)
    
    # Optional: Add slight elevation variation for more dynamic camera movement
    elevation_variation = 5 * np.sin(2 * np.pi * rotation_progress)  # ±5 degree variation
    current_elev = initial_elev + elevation_variation
    
    ax.view_init(elev=current_elev, azim=current_azim)
    
    # Calculate forces
    # 1. Attraction to fixed point
    dir_to_attractor1 = attractor_pos - pos1
    dir_to_attractor1 = tangent_component(dir_to_attractor1, pos1)  # Keep on sphere
    attraction_force1 = attraction_strength * dir_to_attractor1
    
    dir_to_attractor2 = attractor_pos - pos2
    dir_to_attractor2 = tangent_component(dir_to_attractor2, pos2)
    attraction_force2 = attraction_strength * dir_to_attractor2
    
    # 2. Gaussian-based repulsion between dots
    separation = pos2 - pos1
    distance = np.linalg.norm(separation)
    
    # Use Gaussian repulsion force that matches the visual bumps
    repulsion_force1 = gaussian_repulsion_force(pos1, pos2, repulsion_strength, gaussian_sigma)
    repulsion_force2 = gaussian_repulsion_force(pos2, pos1, repulsion_strength, gaussian_sigma)
    
    # Keep forces tangent to sphere
    repulsion_force1 = tangent_component(repulsion_force1, pos1)
    repulsion_force2 = tangent_component(repulsion_force2, pos2)
    
    # Total forces
    total_force1 = attraction_force1 + repulsion_force1
    total_force2 = attraction_force2 + repulsion_force2
    
    # Update velocities
    vel1 += total_force1 * dt
    vel2 += total_force2 * dt
    
    # Keep velocities tangent to sphere
    vel1 = tangent_component(vel1, pos1)
    vel2 = tangent_component(vel2, pos2)
    
    # Apply damping
    vel1 *= damping
    vel2 *= damping
    
    # Update positions
    pos1 += vel1 * dt
    pos2 += vel2 * dt
    
    # Project back to sphere surface
    pos1 = project_to_sphere(pos1)
    pos2 = project_to_sphere(pos2)
    
    # Update dot positions
    dot1.set_data(np.array([pos1[0]]), np.array([pos1[1]]))
    dot1.set_3d_properties(np.array([pos1[2]]))
    
    dot2.set_data(np.array([pos2[0]]), np.array([pos2[1]]))
    dot2.set_3d_properties(np.array([pos2[2]]))
    
    # Update trails - no length limit, keep all history
    trail1_x.append(pos1[0])
    trail1_y.append(pos1[1])
    trail1_z.append(pos1[2])
    
    trail2_x.append(pos2[0])
    trail2_y.append(pos2[1])
    trail2_z.append(pos2[2])
    
    # Update trail lines
    if len(trail1_x) > 1:
        trail_line1.set_data(np.array(trail1_x), np.array(trail1_y))
        trail_line1.set_3d_properties(np.array(trail1_z))
        trail_points1.set_data(np.array(trail1_x[:-1]), np.array(trail1_y[:-1]))
        trail_points1.set_3d_properties(np.array(trail1_z[:-1]))
    
    if len(trail2_x) > 1:
        trail_line2.set_data(np.array(trail2_x), np.array(trail2_y))
        trail_line2.set_3d_properties(np.array(trail2_z))
        trail_points2.set_data(np.array(trail2_x[:-1]), np.array(trail2_y[:-1]))
        trail_points2.set_3d_properties(np.array(trail2_z[:-1]))
    
    # Remove previous Gaussian bumps
    if bump_surface1 is not None:
        bump_surface1.remove()
    if bump_surface2 is not None:
        bump_surface2.remove()
    
    # Create new Gaussian bumps
    bump_x1, bump_y1, bump_z1 = create_gaussian_bump(pos1)
    bump_surface1 = ax.plot_wireframe(bump_x1, bump_y1, bump_z1, 
                                    color='#ff6600', alpha=0.7, linewidth=0.8)
    
    bump_x2, bump_y2, bump_z2 = create_gaussian_bump(pos2)
    bump_surface2 = ax.plot_wireframe(bump_x2, bump_y2, bump_z2, 
                                    color='#ff00ff', alpha=0.7, linewidth=0.8)
    
    # Update status text with physics info
    distance_text = 'SEPARATION: {:.3f}'.format(distance)
    energy_text = 'KINETIC ENERGY: {:.3f}'.format(0.5 * (np.linalg.norm(vel1)**2 + np.linalg.norm(vel2)**2))
    # Show when Gaussian fields are overlapping (significant repulsion)
    overlap_strength = np.exp(-distance**2 / (2 * gaussian_sigma**2))
    overlap_text = 'FIELD OVERLAP: {:.3f}'.format(overlap_strength)
    
    try:
        if hasattr(animate, 'distance_text_obj'):
            animate.distance_text_obj.remove()
        if hasattr(animate, 'energy_text_obj'):
            animate.energy_text_obj.remove()
        if hasattr(animate, 'overlap_text_obj'):
            animate.overlap_text_obj.remove()
    except:
        pass
    
    animate.distance_text_obj = fig.text(0.02, 0.86, distance_text, 
                                       color='#00ff41', fontsize=11, 
                                       fontproperties=custom_font)
    animate.energy_text_obj = fig.text(0.02, 0.83, energy_text, 
                                     color='#00ff41', fontsize=11, 
                                     fontproperties=custom_font)
    # Color-code overlap: red when high, green when low
    overlap_color = '#ff4444' if overlap_strength > 0.5 else '#00ff41'
    animate.overlap_text_obj = fig.text(0.02, 0.80, overlap_text, 
                                      color=overlap_color, fontsize=11, 
                                      fontproperties=custom_font)
    
    return dot1, dot2, trail_line1, trail_line2, trail_points1, trail_points2

# Create animation
print("Creating animation...")
anim = animation.FuncAnimation(fig, animate, frames=total_frames, 
                             interval=interval, blit=False, repeat=False)

plt.tight_layout()

# Save as HD MP4 video
print("Saving HD animation as MP4...")
try:
    # Use FFmpeg writer for high quality MP4
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Physics Simulation'), bitrate=5000)
    
    anim.save('physics_simulation_HD.mp4', writer=writer, dpi=150, 
              savefig_kwargs={'facecolor': '#0a0a0a', 'edgecolor': 'none'})
    print("✓ HD animation saved as 'physics_simulation_HD.mp4'")
    
except Exception as e:
    print(f"FFmpeg not available, trying Pillow for GIF: {e}")
    try:
        # Fallback to high quality GIF
        anim.save('physics_simulation_HD.gif', writer='pillow', fps=fps, dpi=150,
                  savefig_kwargs={'facecolor': '#0a0a0a', 'edgecolor': 'none'})
        print("✓ HD animation saved as 'physics_simulation_HD.gif'")
    except Exception as e2:
        print(f"Error saving animation: {e2}")

print(f"\nAnimation details:")
print(f"Duration: {duration} seconds")
print(f"Frame rate: {fps} FPS") 
print(f"Total frames: {total_frames}")
print(f"Resolution: 16x12 inches at 150 DPI (2400x1800 pixels)")

# Uncomment the line below if you want to also display the animation
# plt.show()

print("Multi-agent physics simulation created and saved!")