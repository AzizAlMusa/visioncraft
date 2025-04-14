import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import random
import math
import time

# Problem parameters
width = 100
height = 100
radius = 20
num_circles = 6

def distance_toroidal(x1, y1, x2, y2, width, height):
    """Calculate the shortest distance between two points on a toroidal surface."""
    dx = min(abs(x1 - x2), width - abs(x1 - x2))
    dy = min(abs(y1 - y2), height - abs(y1 - y2))
    return np.sqrt(dx**2 + dy**2)

def is_point_covered(x, y, centers, radius, width, height):
    """Check if a point is covered by any of the circles on a toroidal surface."""
    for cx, cy in centers:
        if distance_toroidal(x, y, cx, cy, width, height) <= radius:
            return True
    return False

def calculate_coverage(centers, radius, width, height, num_points=1000):
    """Calculate coverage percentage using Monte Carlo sampling."""
    covered_points = 0

    for _ in range(num_points):
        px = random.uniform(0, width)
        py = random.uniform(0, height)

        if is_point_covered(px, py, centers, radius, width, height):
            covered_points += 1

    coverage_percentage = (covered_points / num_points) * 100
    return coverage_percentage

def generate_coverage_heatmap(centers, radius, width, height, grid_size=50):
    """Generate a heatmap showing covered areas."""
    coverage_grid = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            x = width * i / grid_size
            y = height * j / grid_size
            coverage_grid[j, i] = 1 if is_point_covered(x, y, centers, radius, width, height) else 0

    return coverage_grid

def get_circle_overlaps(centers, radius, width, height):
    """Calculate which circles overlap with each other."""
    overlaps = []
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            dist = distance_toroidal(
                centers[i][0], centers[i][1], 
                centers[j][0], centers[j][1], 
                width, height
            )
            if dist < 2 * radius:  # Circles overlap
                overlaps.append((i, j, dist))
    return overlaps

def improved_simulated_annealing(initial_centers, radius, width, height, max_iterations=1000,
                           initial_temp=100.0, cooling_rate=0.95, step_size=10.0,
                           min_step_size=0.1, plateau_length=50,
                           reheating_interval=200, reheating_factor=1.5,
                           animation_callback=None):
    """
    Optimize circle placement using simulated annealing with reheating.

    Parameters:
    - initial_centers: Starting circle centers
    - radius: Circle radius
    - width, height: Dimensions of the toroidal plane
    - max_iterations: Maximum number of iterations
    - initial_temp: Starting temperature
    - cooling_rate: Rate at which temperature decreases
    - step_size: Initial maximum displacement size
    - min_step_size: Minimum step size before reducing temperature
    - plateau_length: Number of iterations at each temperature
    - reheating_interval: Frequency of reheating the system
    - reheating_factor: Factor by which to increase temperature during reheating
    - animation_callback: Function to call for animation updates

    Returns:
    - best_centers: Optimized circle centers
    - history: List of (centers, coverage) tuples for animation
    """
    # Initialize variables
    current_centers = initial_centers.copy()
    current_coverage = calculate_coverage(current_centers, radius, width, height)

    best_centers = current_centers.copy()
    best_coverage = current_coverage

    temp = initial_temp
    current_step_size = step_size

    # History for animation
    history = [(current_centers.copy(), current_coverage, 0)]

    # Variables to track progress
    iteration = 0
    plateau_counter = 0
    no_improvement_counter = 0
    reheating_counter = 0

    # Main simulated annealing loop
    while iteration < max_iterations and temp > 0.1:
        # Reduce step size if no recent improvements
        if no_improvement_counter > plateau_length:
            current_step_size = max(min_step_size, current_step_size * 0.8)
            no_improvement_counter = 0
        
        # Periodic reheating to escape local optima
        if iteration > 0 and iteration % reheating_interval == 0:
            reheating_counter += 1
            old_temp = temp
            temp = min(initial_temp, temp * reheating_factor)
            print(f"Reheating {reheating_counter} at iteration {iteration}: {old_temp:.2f} -> {temp:.2f}")
            # Increase step size temporarily to explore more of the space
            current_step_size = min(width/5, current_step_size * 1.5)

        # Create a neighbor solution with improved neighbor generation
        neighbor_centers = current_centers.copy()
        
        # Strategy selection: bias toward moving overlapping circles
        if random.random() < 0.75:  # 75% chance to prioritize overlapping circles
            # Find circles with overlaps
            overlaps = get_circle_overlaps(current_centers, radius, width, height)
            
            if overlaps:
                # Count overlaps per circle
                overlap_counts = {}
                for i, j, _ in overlaps:
                    overlap_counts[i] = overlap_counts.get(i, 0) + 1
                    overlap_counts[j] = overlap_counts.get(j, 0) + 1
                
                # Select circle with more overlaps
                if random.random() < 0.7 and overlap_counts:  # Mostly choose most overlapping
                    circle_idx = max(overlap_counts.items(), key=lambda x: x[1])[0]
                else:  # Sometimes select from overlapping circles randomly
                    overlap_circles = list(overlap_counts.keys())
                    circle_idx = random.choice(overlap_circles) if overlap_circles else random.randint(0, len(neighbor_centers) - 1)
            else:
                circle_idx = random.randint(0, len(neighbor_centers) - 1)
        else:
            # Standard random selection
            circle_idx = random.randint(0, len(neighbor_centers) - 1)
        
        # Random displacement with direction bias
        # Check if this circle overlaps with others
        is_overlapping = False
        for i, j, _ in get_circle_overlaps(current_centers, radius, width, height):
            if circle_idx == i or circle_idx == j:
                is_overlapping = True
                break
        
        # Generate displacement with bias if overlapping
        if is_overlapping and random.random() < 0.6:
            # Find direction away from overlap center
            cx, cy = current_centers[circle_idx]
            overlap_x, overlap_y = 0, 0
            count = 0
            
            for i, j, _ in get_circle_overlaps(current_centers, radius, width, height):
                if circle_idx == i:
                    overlap_x += current_centers[j][0]
                    overlap_y += current_centers[j][1]
                    count += 1
                elif circle_idx == j:
                    overlap_x += current_centers[i][0]
                    overlap_y += current_centers[i][1]
                    count += 1
            
            if count > 0:
                overlap_x /= count
                overlap_y /= count
                
                # Calculate direction away from overlaps
                dx = cx - overlap_x
                dy = cy - overlap_y
                
                # Normalize and scale by step size with some randomness
                mag = np.sqrt(dx**2 + dy**2)
                if mag > 0:
                    dx = dx / mag * current_step_size * random.uniform(0.5, 1.5)
                    dy = dy / mag * current_step_size * random.uniform(0.5, 1.5)
                else:
                    dx = random.uniform(-current_step_size, current_step_size)
                    dy = random.uniform(-current_step_size, current_step_size)
            else:
                dx = random.uniform(-current_step_size, current_step_size)
                dy = random.uniform(-current_step_size, current_step_size)
        else:
            # Standard random displacement
            dx = random.uniform(-current_step_size, current_step_size)
            dy = random.uniform(-current_step_size, current_step_size)

        # Apply displacement with toroidal wrapping
        new_x = (neighbor_centers[circle_idx][0] + dx) % width
        new_y = (neighbor_centers[circle_idx][1] + dy) % height

        neighbor_centers[circle_idx] = (new_x, new_y)

        # Calculate new coverage
        neighbor_coverage = calculate_coverage(neighbor_centers, radius, width, height)

        # Decide whether to accept the new solution
        delta_coverage = neighbor_coverage - current_coverage

        # Modified acceptance probability - more eager at high temperatures
        # and after reheating events
        accept_prob = np.exp(delta_coverage / temp)
        if reheating_counter > 0 and iteration % reheating_interval < 50:
            # More exploration after reheating
            accept_prob *= 1.2
            
        # Always accept if better, sometimes accept if worse (based on temperature)
        if delta_coverage > 0 or random.random() < accept_prob:
            current_centers = neighbor_centers.copy()
            current_coverage = neighbor_coverage

            # Update best solution if improved
            if current_coverage > best_coverage:
                best_centers = current_centers.copy()
                best_coverage = current_coverage
                no_improvement_counter = 0
                print(f"New best at iteration {iteration}: Coverage={best_coverage:.2f}%")
            else:
                no_improvement_counter += 1
        else:
            no_improvement_counter += 1

        # Increment plateau counter
        plateau_counter += 1

        # Cool down temperature after plateau_length iterations
        if plateau_counter >= plateau_length:
            temp *= cooling_rate
            plateau_counter = 0
            print(f"Iteration {iteration}, Temperature: {temp:.2f}, "
                  f"Best Coverage: {best_coverage:.2f}%, "
                  f"Current Step Size: {current_step_size:.2f}")

        # Record history for animation
        if animation_callback:
            animation_callback(current_centers.copy(), current_coverage, iteration)

        history.append((current_centers.copy(), current_coverage, iteration))
        iteration += 1

    print(f"Simulated Annealing completed after {iteration} iterations")
    print(f"Best Coverage: {best_coverage:.2f}%")

    return best_centers, history

def generate_improved_initial_state(num_circles, radius, width, height):
    """Generate an improved initial state with better circle placement."""
    # Approach 1: Try to distribute circles more evenly
    best_centers = []
    best_coverage = 0
    best_min_distance = 0
    
    # Try a few different initial states and pick the best one
    num_attempts = 5
    
    for attempt in range(num_attempts):
        if attempt == 0:
            # First attempt: Clustered near the center
            cluster_center = (width / 2, height / 2)
            cluster_spread = radius * 1.5  # Tighter cluster
            centers = []
            for _ in range(num_circles):
                x = np.clip(random.gauss(cluster_center[0], cluster_spread), 0, width)
                y = np.clip(random.gauss(cluster_center[1], cluster_spread), 0, height)
                centers.append((x, y))
        
        elif attempt == 1:
            # Second attempt: Placed along a circle
            centers = []
            circle_radius = min(width, height) / 3
            center_x, center_y = width / 2, height / 2
            
            for i in range(num_circles):
                angle = 2 * np.pi * i / num_circles
                x = center_x + circle_radius * np.cos(angle)
                y = center_y + circle_radius * np.sin(angle)
                centers.append((x, y))
        
        elif attempt == 2:
            # Third attempt: Grid-like pattern with some randomness
            grid_size = int(np.ceil(np.sqrt(num_circles)))
            cell_width = width / grid_size
            cell_height = height / grid_size
            
            centers = []
            for i in range(num_circles):
                row = i // grid_size
                col = i % grid_size
                
                x = (col + 0.5) * cell_width + random.uniform(-cell_width/4, cell_width/4)
                y = (row + 0.5) * cell_height + random.uniform(-cell_height/4, cell_height/4)
                
                centers.append((x, y))
        
        else:
            # Other attempts: Random placement
            centers = []
            for _ in range(num_circles):
                x = random.uniform(0, width)
                y = random.uniform(0, height)
                centers.append((x, y))
        
        # Evaluate this initial state
        coverage = calculate_coverage(centers, radius, width, height)
        
        # Calculate minimum distance between any two circles
        min_distance = float('inf')
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dist = distance_toroidal(
                    centers[i][0], centers[i][1],
                    centers[j][0], centers[j][1],
                    width, height
                )
                min_distance = min(min_distance, dist)
        
        # Score based on both coverage and distribution
        # We want high coverage but also good spacing between circles
        score = coverage
        
        if score > best_coverage:
            best_centers = centers
            best_coverage = coverage
            best_min_distance = min_distance
    
    print(f"Selected initial state with coverage: {best_coverage:.2f}%, "
          f"minimum circle distance: {best_min_distance:.2f}")
    
    return best_centers

def animate_simulated_annealing():
    """Run simulated annealing with animation."""
    # Generate improved initial state
    initial_centers = generate_improved_initial_state(num_circles, radius, width, height)

    # Set up the figure for animation
    fig = plt.figure(figsize=(16, 8))

    # Left subplot for circle positions
    ax1 = fig.add_subplot(121)
    ax1.set_xlim(-5, width+5)
    ax1.set_ylim(-5, height+5)
    ax1.set_aspect('equal')
    ax1.set_title("Circle Positions")
    ax1.set_xlabel("X coordinate")
    ax1.set_ylabel("Y coordinate")
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Draw the boundary box
    boundary = plt.Rectangle((0, 0), width, height, fill=False, edgecolor='black', linewidth=2)
    ax1.add_patch(boundary)

    # Initialize circle patches
    circles = []
    for i in range(num_circles):
        circle = Circle(initial_centers[i], radius, fill=False, edgecolor='blue', linewidth=2)
        ax1.add_patch(circle)
        circles.append(circle)

    # Initialize center point markers
    centers, = ax1.plot([c[0] for c in initial_centers], [c[1] for c in initial_centers], 'ro', markersize=6)

    # Custom colormap for heatmap
    custom_cmap = LinearSegmentedColormap.from_list('RdGn', ['red', 'green'])

    # Right subplot for coverage heatmap
    ax2 = fig.add_subplot(122)
    initial_heatmap = generate_coverage_heatmap(initial_centers, radius, width, height)
    heatmap = ax2.imshow(initial_heatmap, extent=[0, width, 0, height], origin='lower',
                        cmap=custom_cmap, vmin=0, vmax=1, alpha=0.8)
    plt.colorbar(heatmap, ax=ax2, label='Coverage')
    ax2.set_title(f"Coverage Heatmap (Initial: {calculate_coverage(initial_centers, radius, width, height):.2f}%)")
    ax2.set_xlabel("X coordinate")
    ax2.set_ylabel("Y coordinate")

    # Initialize container for history
    history = []

    # Add plot for coverage progress
    ax3 = fig.add_axes([0.15, 0.01, 0.7, 0.15])  # [left, bottom, width, height]
    ax3.set_xlim(0, 1000)  # Will adjust later based on actual iterations
    ax3.set_ylim(0, 100)
    ax3.set_title("Coverage Progress")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Coverage %")
    coverage_line, = ax3.plot([], [], 'b-')

    # Add best coverage text
    best_text = ax1.text(5, height - 5, "", fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    # Animation callback function
    def update_animation(centers_update, coverage_update, iteration):
        history.append((centers_update, coverage_update, iteration))

    # Run improved simulated annealing
    print("Starting improved simulated annealing with reheating...")
    best_centers, _ = improved_simulated_annealing(
        initial_centers, radius, width, height,
        max_iterations=1000,
        initial_temp=100.0,
        cooling_rate=0.95,
        step_size=10.0,
        min_step_size=0.1,
        plateau_length=50,
        reheating_interval=200,  # Reheat every 200 iterations
        reheating_factor=1.5,    # Increase temperature by 50%
        animation_callback=update_animation
    )

    # Adjust x-axis limit based on actual iterations
    ax3.set_xlim(0, len(history))

    # Animation update function
    def update(frame):
        if frame < len(history):
            current_centers, current_coverage, iteration = history[frame]

            # Update circle positions
            for i in range(num_circles):
                circles[i].center = current_centers[i]

            # Update center points
            centers.set_data([c[0] for c in current_centers], [c[1] for c in current_centers])

            # Update heatmap
            current_heatmap = generate_coverage_heatmap(current_centers, radius, width, height)
            heatmap.set_array(current_heatmap)
            ax2.set_title(f"Coverage Heatmap (Current: {current_coverage:.2f}%)")

            # Update coverage progress plot with color gradient
            iterations = [h[2] for h in history[:frame+1]]
            coverages = [h[1] for h in history[:frame+1]]
            
            # Clear previous line
            for collection in ax3.collections:
                collection.remove()
                
            # Draw colored segments
            cmap = plt.cm.jet
            for i in range(len(iterations)-1):
                color = cmap(i / max(1, len(iterations)-2))
                ax3.plot([iterations[i], iterations[i+1]], 
                         [coverages[i], coverages[i+1]], 
                         color=color, 
                         linewidth=1.5)
            
            # Still update original line data, but make it invisible
            coverage_line.set_data(iterations, coverages)
            coverage_line.set_visible(False)

            # Update best text
            best_so_far = max(coverages)
            best_text.set_text(f"Best Coverage: {best_so_far:.2f}%\nIteration: {iteration}")

            # Update figure title
            fig.suptitle(f"Improved Simulated Annealing for Circle Coverage - Iteration {iteration}", fontsize=16)

        return circles + [centers, heatmap, coverage_line, best_text]

    # Create animation
    frames = len(history)
    print(f"Creating animation with {frames} frames...")

    anim = animation.FuncAnimation(
        fig, update, frames=frames, interval=50, blit=True
    )

    # Save animation
    writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    anim.save("circle_coverage_annealing.mp4", writer=writer)

    # Display final result
    print("Animation saved as 'circle_coverage_annealing.mp4'")
    print(f"Final Best Coverage: {calculate_coverage(best_centers, radius, width, height, num_points=10000):.2f}%")

    # Also create a static figure of the final result
    plot_final_result(best_centers, radius, width, height)

    return best_centers

def plot_final_result(centers, radius, width, height):
    """Create a static visualization of the final optimized arrangement."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Calculate coverage
    coverage = calculate_coverage(centers, radius, width, height, num_points=10000)

    # Generate and display coverage heatmap
    heatmap = generate_coverage_heatmap(centers, radius, width, height, grid_size=100)
    custom_cmap = LinearSegmentedColormap.from_list('RdGn', ['red', 'green'])
    im = ax.imshow(heatmap, extent=[0, width, 0, height], origin='lower',
                  cmap=custom_cmap, vmin=0, vmax=1, alpha=0.7)
    plt.colorbar(im, ax=ax, label='Coverage')

    # Draw plane boundary
    ax.add_patch(plt.Rectangle((0, 0), width, height, fill=False, edgecolor='black', linewidth=2))

    # Draw circles
    for i, (cx, cy) in enumerate(centers):
        circle = Circle((cx, cy), radius, fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(circle)

        # Mark center points
        ax.plot(cx, cy, 'ro', markersize=6)

        # Label circles
        ax.text(cx, cy, str(i+1), ha='center', va='center', fontweight='bold', color='white',
               bbox=dict(facecolor='blue', alpha=0.7, edgecolor='none', pad=3))

        # Draw extended circles at borders to show toroidal wrapping
        # Left edge
        if cx < radius:
            wrap_circle = Circle((cx + width, cy), radius, fill=False, edgecolor='blue',
                               linewidth=1, linestyle='--')
            ax.add_patch(wrap_circle)
        # Right edge
        if cx > width - radius:
            wrap_circle = Circle((cx - width, cy), radius, fill=False, edgecolor='blue',
                               linewidth=1, linestyle='--')
            ax.add_patch(wrap_circle)
        # Bottom edge
        if cy < radius:
            wrap_circle = Circle((cx, cy + height), radius, fill=False, edgecolor='blue',
                               linewidth=1, linestyle='--')
            ax.add_patch(wrap_circle)
        # Top edge
        if cy > height - radius:
            wrap_circle = Circle((cx, cy - height), radius, fill=False, edgecolor='blue',
                               linewidth=1, linestyle='--')
            ax.add_patch(wrap_circle)

    # Set plot properties
    ax.set_xlim(-5, width+5)
    ax.set_ylim(-5, height+5)
    ax.set_aspect('equal')
    ax.set_title(f"Optimized Circle Arrangement using Improved Simulated Annealing\nCoverage: {coverage:.2f}%", fontsize=14)
    ax.set_xlabel("X coordinate", fontsize=12)
    ax.set_ylabel("Y coordinate", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add text with circle coordinates

    plt.tight_layout()
    plt.savefig("optimized_circle_arrangement.png", dpi=300)

    # Calculate theoretical upper bound
    circle_area = np.pi * radius**2
    total_area = width * height
    max_theoretical_coverage = min(100, (num_circles * circle_area / total_area) * 100)

    print(f"Theoretical upper bound (with no overlap): {max_theoretical_coverage:.2f}%")
    print(f"Achieved coverage: {coverage:.2f}%")
    print(f"Efficiency: {(coverage / max_theoretical_coverage) * 100:.2f}%")

    plt.show()

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Run simulated annealing with animation
    best_centers = animate_simulated_annealing()

    # Print final best centers
    print("\nOptimized Circle Centers:")
    for i, (cx, cy) in enumerate(best_centers):
        print(f"Circle {i+1}: ({cx:.2f}, {cy:.2f})")