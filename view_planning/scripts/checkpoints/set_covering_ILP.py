import numpy as np
import matplotlib.pyplot as plt
import pulp
from scipy.spatial import distance

# Parameters
width, height = 100, 100  # Size of the plane
num_points = 1000  # Number of sampled points
radius = 20  # Radius of the circle
min_dist = 2  # Minimum distance between points (Poisson disk sampling)

# Poisson Disk Sampling: Ensure minimum spacing between points
def poisson_disk_sampling(width, height, num_points, min_dist):
    points = []
    while len(points) < num_points:
        x, y = np.random.uniform(0, width), np.random.uniform(0, height)
        if all(np.linalg.norm(np.array([x, y]) - np.array(p)) >= min_dist for p in points):
            points.append((x, y))
    return np.array(points)

# Generate sampled points
points = poisson_disk_sampling(width, height, num_points, min_dist)

# Define possible circle centers (same as points to avoid redundancy)
circle_centers = points.copy()
num_circles = len(circle_centers)

# Compute coverage matrix (binary matrix where M[i][j] = 1 if circle i covers point j)
coverage_matrix = np.zeros((num_circles, num_points), dtype=int)

for i, (cx, cy) in enumerate(circle_centers):
    for j, (px, py) in enumerate(points):
        if np.linalg.norm(np.array([cx, cy]) - np.array([px, py])) <= radius:
            coverage_matrix[i, j] = 1

# ILP Problem: Minimize the number of circles while covering all points
prob = pulp.LpProblem("Set_Cover_Problem", pulp.LpMinimize)

# Decision variables (1 if circle i is selected, 0 otherwise)
circle_vars = [pulp.LpVariable(f"circle_{i}", cat="Binary") for i in range(num_circles)]

# Objective function: Minimize number of circles
prob += pulp.lpSum(circle_vars)

# Constraints: Each point must be covered by at least one circle
for j in range(num_points):
    prob += pulp.lpSum(circle_vars[i] * coverage_matrix[i, j] for i in range(num_circles)) >= 1

# Solve ILP
prob.solve()

# Get selected circles
selected_circles = [i for i in range(num_circles) if pulp.value(circle_vars[i]) == 1]

# Visualization
fig, ax = plt.subplots(figsize=(8, 8))

# Plot points
ax.scatter(points[:, 0], points[:, 1], color='blue', label='Points', alpha=0.6)

# Plot circles
for i in selected_circles:
    cx, cy = circle_centers[i]
    circle = plt.Circle((cx, cy), radius, color='red', fill=True, alpha=0.3, linewidth=2)
    ax.add_artist(circle)

ax.set_xlim(0, width)
ax.set_ylim(0, height)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title(f'ILP Coverage with {len(selected_circles)} Circles (Optimized)')

plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.show()

# Report
print(f"Final number of circles (ILP Optimized): {len(selected_circles)}")
