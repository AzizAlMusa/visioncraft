import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
num_viewpoints = 20            # Number of viewpoints
circle_radius = 5.0            # Radius of the circle
k_attr = 1.0                   # Attraction constant
k_repel_obj = 0.0              # Repulsion from covered areas (set to zero if only attraction)
k_repel_vp = 0.1              # Repulsion between viewpoints
delta_t = 0.1                  # Time step
total_time = 20                # Total simulation time
num_steps = int(total_time / delta_t)

# Initialize viewpoints at random angles
np.random.seed(42)
theta = np.random.uniform(0, 1e-03, num_viewpoints)
theta = np.sort(theta)  # Sort for better visualization

# Initialize velocities
omega = np.zeros(num_viewpoints)

# Initialize visibility function V(theta)
# For simplicity, let's define V as a function with uncovered regions
# Let's assume uncovered regions at specific angles
uncovered_regions = [
    (np.pi / 4, np.pi / 2),
    (3 * np.pi / 2, 7 * np.pi / 4)
]

def visibility_function(theta):
    V = np.ones_like(theta)
    for start, end in uncovered_regions:
        indices = (theta >= start) & (theta <= end)
        V[indices] = 0.5  # Lower visibility in uncovered regions
    return V

def dV_dtheta(theta):
    # For simplicity, approximate derivative numerically
    delta = 1e-5
    V_plus = visibility_function(theta + delta)
    V_minus = visibility_function(theta - delta)
    return (V_plus - V_minus) / (2 * delta)

# Prepare for animation
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-circle_radius - 1, circle_radius + 1)
ax.set_ylim(-circle_radius - 1, circle_radius + 1)
ax.set_aspect('equal')
circle = plt.Circle((0, 0), circle_radius, color='black', fill=False)
ax.add_artist(circle)
viewpoints_plot, = ax.plot([], [], 'bo', ms=8)
time_template = 'Time = %.1f s'
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)
lines = []

# Plot uncovered regions
for start, end in uncovered_regions:
    theta_region = np.linspace(start, end, 100)
    x_region = circle_radius * np.cos(theta_region)
    y_region = circle_radius * np.sin(theta_region)
    ax.plot(x_region, y_region, 'r', linewidth=5, alpha=0.3)

def update(frame):
    global theta, omega

    # Compute forces
    V = visibility_function(theta)
    dV = dV_dtheta(theta)
    F_attr = -k_attr * dV
    F_repel_obj = k_repel_obj * dV  # Set to zero if not used

    # Repulsion between viewpoints
    F_repel_vp = np.zeros_like(theta)
    for i in range(num_viewpoints):
        for j in range(num_viewpoints):
            if i != j:
                # Compute angular distance
                delta_theta = theta[i] - theta[j]
                # Adjust for circular topology
                delta_theta = np.arctan2(np.sin(delta_theta), np.cos(delta_theta))
                distance = np.abs(delta_theta)
                if distance < np.pi:
                    F_repel_vp[i] += k_repel_vp * delta_theta / (distance**2 + 1e-5)

    # Total force
    F_total = F_attr + F_repel_obj + F_repel_vp

    # Update angular velocities and positions
    omega += F_total * delta_t
    omega *= 0.9  # Damping
    theta += omega * delta_t

    # Keep theta within [0, 2*pi]
    theta = np.mod(theta, 2 * np.pi)

    # Update plot
    x = circle_radius * np.cos(theta)
    y = circle_radius * np.sin(theta)
    viewpoints_plot.set_data(x, y)
    time_text.set_text(time_template % (frame * delta_t))

    return viewpoints_plot, time_text

# Create the animation
ani = FuncAnimation(fig, update, frames=num_steps, interval=50, blit=True)

plt.title('Physics-Inspired Viewpoint Distribution on a Circle')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
