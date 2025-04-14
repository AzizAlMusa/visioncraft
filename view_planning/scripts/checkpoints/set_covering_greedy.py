import numpy as np
import matplotlib.pyplot as plt
import random

# Parameters
width, height = 100, 100  # Size of the plane
num_points = 1000  # Number of points to sample
radius = 20  # Radius of the circle
min_dist = 2  # Minimum distance between points for Poisson disk sampling

# Poisson disk sampling function
def poisson_disk_sampling(width, height, num_points, min_dist):
    grid_size = min_dist / np.sqrt(2)  # Grid cell size
    grid_width = int(np.ceil(width / grid_size))
    grid_height = int(np.ceil(height / grid_size))
    
    grid = np.empty((grid_width, grid_height), dtype=object)
    points = []
    active_list = []
    
    def get_cell(x, y):
        return int(x // grid_size), int(y // grid_size)
    
    def is_valid_point(x, y):
        cell_x, cell_y = get_cell(x, y)
        # Check the neighboring cells for validity
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = cell_x + dx, cell_y + dy
                if 0 <= nx < grid_width and 0 <= ny < grid_height:
                    if grid[nx, ny]:
                        for px, py in grid[nx, ny]:
                            if np.sqrt((x - px) ** 2 + (y - py) ** 2) < min_dist:
                                return False
        return True

    # Start with a random point
    first_point = (random.uniform(0, width), random.uniform(0, height))
    points.append(first_point)
    active_list.append(first_point)
    cell_x, cell_y = get_cell(*first_point)
    grid[cell_x, cell_y] = [first_point]
    
    while active_list:
        point = random.choice(active_list)
        found = False
        for _ in range(30):  # Attempt to generate 30 new points per active point
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(min_dist, 2 * min_dist)
            new_point = (point[0] + distance * np.cos(angle), point[1] + distance * np.sin(angle))
            
            # Check if the point is within bounds and valid
            if 0 <= new_point[0] < width and 0 <= new_point[1] < height and is_valid_point(*new_point):
                points.append(new_point)
                active_list.append(new_point)
                cell_x, cell_y = get_cell(*new_point)
                if grid[cell_x, cell_y] is None:
                    grid[cell_x, cell_y] = []
                grid[cell_x, cell_y].append(new_point)
                found = True
                break
        
        if not found:
            active_list.remove(point)
    
    return np.array(points)

# Generate uniformly distributed points
points = poisson_disk_sampling(width, height, num_points, min_dist)

uncovered_points = set(range(len(points)))  # All points start as uncovered
circles = []

# Function to compute toroidal distance
def toroidal_distance(x1, y1, x2, y2, width, height):
    dx = min(abs(x1 - x2), width - abs(x1 - x2))  # Wrap around x-axis
    dy = min(abs(y1 - y2), height - abs(y1 - y2))  # Wrap around y-axis
    return np.sqrt(dx**2 + dy**2)

# Function to check if a point is inside a toroidal circle
def is_inside_circle(x, y, cx, cy, r):
    return toroidal_distance(x, y, cx, cy, width, height) <= r

# Function to find the best circle to place (covers the most uncovered points)
def find_best_circle():
    best_circle = None
    max_covered = 0
    best_covered_indices = set()

    # Sample candidate circle centers from uncovered points
    candidate_centers = points[list(uncovered_points)]

    for cx, cy in candidate_centers:
        covered_indices = {i for i in uncovered_points if is_inside_circle(points[i, 0], points[i, 1], cx, cy, radius)}
        
        if len(covered_indices) > max_covered:
            max_covered = len(covered_indices)
            best_circle = (cx, cy)
            best_covered_indices = covered_indices

    return best_circle, best_covered_indices

# Greedy coverage loop
while uncovered_points:
    best_circle, covered_indices = find_best_circle()
    
    if not best_circle:
        break  # If no valid circles can be placed, break

    # Add the circle and mark points as covered
    circles.append(best_circle)
    uncovered_points -= covered_indices

# Visualization with toroidal wrap-around effects
fig, ax = plt.subplots(figsize=(8, 8))

# Plot circles with shading and overlaps
for cx, cy in circles:
    circle = plt.Circle((cx, cy), radius, color='blue', fill=True, alpha=0.3, linewidth=2)
    ax.add_artist(circle)
    
    # Wrap around: duplicate circles across edges
    if cx - radius < 0:  # Left edge wrap-around
        ax.add_artist(plt.Circle((cx + width, cy), radius, color='blue', fill=True, alpha=0.3))
    if cx + radius > width:  # Right edge wrap-around
        ax.add_artist(plt.Circle((cx - width, cy), radius, color='blue', fill=True, alpha=0.3))
    if cy - radius < 0:  # Bottom edge wrap-around
        ax.add_artist(plt.Circle((cx, cy + height), radius, color='blue', fill=True, alpha=0.3))
    if cy + radius > height:  # Top edge wrap-around
        ax.add_artist(plt.Circle((cx, cy - height), radius, color='blue', fill=True, alpha=0.3))
    
    # Wrap at corners
    if cx - radius < 0 and cy - radius < 0:  # Bottom-left corner
        ax.add_artist(plt.Circle((cx + width, cy + height), radius, color='blue', fill=True, alpha=0.3))
    if cx + radius > width and cy - radius < 0:  # Bottom-right corner
        ax.add_artist(plt.Circle((cx - width, cy + height), radius, color='blue', fill=True, alpha=0.3))
    if cx - radius < 0 and cy + radius > height:  # Top-left corner
        ax.add_artist(plt.Circle((cx + width, cy - height), radius, color='blue', fill=True, alpha=0.3))
    if cx + radius > width and cy + radius > height:  # Top-right corner
        ax.add_artist(plt.Circle((cx - width, cy - height), radius, color='blue', fill=True, alpha=0.3))

ax.set_xlim(0, width)
ax.set_ylim(0, height)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title(f'Coverage with {len(circles)} Circles (Toroidal Boundaries)')

plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# Report
print(f"Final number of circles: {len(circles)}")
print(f"Final coverage: {100 * (1 - len(uncovered_points) / num_points):.2f}%")
