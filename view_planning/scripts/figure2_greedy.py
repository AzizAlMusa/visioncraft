import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import os
import time

# Try to import numba for JIT compilation (optional but recommended)
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorators that do nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(x):
        return range(x)

# === Arguments ===
parser = argparse.ArgumentParser()
parser.add_argument("--strategy", type=str, default="greedy", choices=["greedy", "random"])
parser.add_argument("--potential_type", type=str, default="log")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save_dir", type=str, default="./results2")
parser.add_argument("--animate", action="store_true")
parser.add_argument("--k_attr", type=float, default=0.4)
parser.add_argument("--k_rep", type=float, default=1.0)
parser.add_argument("--verbose", action="store_true")
# Greedy specific arguments
parser.add_argument("--candidate_density", type=int, default=5, help="Density of candidate viewpoints per iteration")
parser.add_argument("--coverage_threshold", type=float, default=0.25, help="Threshold for considering a point covered")
args = parser.parse_args()

# === Constants ===
np.random.seed(args.seed)
grid_size = 100
fov_radius = 20
epsilon = 1e-6
max_viewpoints = 100

x = np.arange(grid_size)
y = np.arange(grid_size)
X, Y = np.meshgrid(x, y)
field_points = np.stack([X.ravel(), Y.ravel()], axis=-1)

if args.verbose:
    print(f"Numba JIT available: {NUMBA_AVAILABLE}")

# === Optimized helper functions ===
@jit(nopython=True, parallel=True)
def fast_wrap_distance(diff, grid_size):
    """Optimized distance wrapping with Numba"""
    result = np.empty_like(diff)
    for i in prange(diff.shape[0]):
        for j in range(diff.shape[1]):
            for k in range(diff.shape[2]):
                val = diff[i, j, k]
                if abs(val) > grid_size / 2:
                    result[i, j, k] = -np.sign(val) * (grid_size - abs(val))
                else:
                    result[i, j, k] = val
    return result

@jit(nopython=True, parallel=True)
def fast_visibility_computation(field_points, particles, grid_size, fov_radius, beta=20.0):
    """Optimized visibility computation"""
    n_points = field_points.shape[0]
    n_particles = particles.shape[0]
    visibility = np.zeros(n_points)
    
    for i in prange(n_points):
        total_vis = 0.0
        for j in range(n_particles):
            # Compute wrapped difference
            dx = field_points[i, 0] - particles[j, 0]
            dy = field_points[i, 1] - particles[j, 1]
            
            # Apply wrapping
            if abs(dx) > grid_size / 2:
                dx = -np.sign(dx) * (grid_size - abs(dx))
            if abs(dy) > grid_size / 2:
                dy = -np.sign(dy) * (grid_size - abs(dy))
                
            distance = np.sqrt(dx*dx + dy*dy) + 1e-6
            s = 1.0 / (1.0 + np.exp(beta * (distance / fov_radius - 1.0)))
            total_vis += s
            
        visibility[i] = min(total_vis, 1.0)
    
    return visibility

@jit(nopython=True)
def compute_coverage_matrix(field_points, candidates, grid_size, fov_radius, beta=20.0):
    """Compute which field points each candidate can cover"""
    n_points = field_points.shape[0]
    n_candidates = candidates.shape[0]
    coverage_matrix = np.zeros((n_candidates, n_points))
    
    for j in range(n_candidates):
        for i in range(n_points):
            # Compute wrapped difference
            dx = field_points[i, 0] - candidates[j, 0]
            dy = field_points[i, 1] - candidates[j, 1]
            
            # Apply wrapping
            if abs(dx) > grid_size / 2:
                dx = -np.sign(dx) * (grid_size - abs(dx))
            if abs(dy) > grid_size / 2:
                dy = -np.sign(dy) * (grid_size - abs(dy))
                
            distance = np.sqrt(dx*dx + dy*dy) + 1e-6
            s = 1.0 / (1.0 + np.exp(beta * (distance / fov_radius - 1.0)))
            coverage_matrix[j, i] = s
    
    return coverage_matrix

# === Greedy Coverage Class ===
class GreedyCoverage:
    def __init__(self, grid_size, fov_radius, use_wrapping=True):
        self.grid_size = grid_size
        self.fov_radius = fov_radius
        self.use_wrapping = use_wrapping
        self.field_points = field_points
        self.coverage_threshold = args.coverage_threshold
        
    def wrap_distance(self, diff):
        if not self.use_wrapping:
            return diff
        
        if NUMBA_AVAILABLE and diff.ndim == 3:
            return fast_wrap_distance(diff, self.grid_size)
        else:
            return np.where(np.abs(diff) > self.grid_size / 2,
                            -np.sign(diff) * (self.grid_size - np.abs(diff)), diff)

    def generate_candidate_viewpoints(self, n_candidates, coverage_weights=None):
        """Generate candidate viewpoints, biased towards uncovered areas if weights provided"""
        if coverage_weights is None:
            # Random sampling
            candidates = np.random.rand(n_candidates, 2) * self.grid_size
        else:
            # Weighted sampling based on coverage needs
            # Higher weights for less covered areas
            weights = 1.0 - coverage_weights + epsilon
            weights = weights / np.sum(weights)
            
            # Sample field points based on weights
            indices = np.random.choice(len(self.field_points), 
                                       size=min(n_candidates, len(self.field_points)), 
                                       p=weights, replace=False)
            
            # Generate candidates around selected points
            candidates = []
            for idx in indices:
                base_point = self.field_points[idx]
                # Add some noise around the point
                noise = np.random.normal(0, fov_radius * 0.3, 2)
                candidate = base_point + noise
                # Wrap around boundaries
                candidate = candidate % self.grid_size
                candidates.append(candidate)
            
            candidates = np.array(candidates)
            
            # Fill remaining candidates with random points if needed
            if len(candidates) < n_candidates:
                remaining = n_candidates - len(candidates)
                random_candidates = np.random.rand(remaining, 2) * self.grid_size
                candidates = np.vstack([candidates, random_candidates])
                
        return candidates

    def compute_visibility(self, viewpoint, field_points=None):
        """Compute visibility from a single viewpoint"""
        if field_points is None:
            field_points = self.field_points
            
        diff = field_points - viewpoint[None, :]
        wrapped_diff = self.wrap_distance(diff[:, None, :])
        distances = np.linalg.norm(wrapped_diff.squeeze(), axis=-1) + epsilon
        
        # Smooth visibility function
        beta = 20.0
        visibility = 1.0 / (1.0 + np.exp(beta * (distances / self.fov_radius - 1.0)))
        
        return visibility

    def greedy_set_cover(self, max_viewpoints=None, target_coverage=0.99):
        """
        Greedy set covering algorithm adapted from Scott 2009 paper
        """
        if max_viewpoints is None:
            max_viewpoints = max_viewpoints
            
        selected_viewpoints = []
        coverage = np.zeros(len(self.field_points))
        total_coverage_history = []
        
        iteration = 0
        start_time = time.time()
        
        while len(selected_viewpoints) < max_viewpoints:
            # Check if we've achieved target coverage
            covered_ratio = np.mean(coverage > self.coverage_threshold)
            total_coverage_history.append(covered_ratio)
            
            if args.verbose:
                print(f"[Greedy Iteration {iteration:02d}] Coverage = {covered_ratio:.4f} | Viewpoints = {len(selected_viewpoints)}")
            
            if covered_ratio >= target_coverage:
                break
                
            # Generate candidate viewpoints
            # Bias towards uncovered areas
            n_candidates = args.candidate_density * 20  # More candidates per iteration
            candidates = self.generate_candidate_viewpoints(n_candidates, coverage)
            
            # Find the best candidate (one that covers most uncovered points)
            best_candidate = None
            best_score = -1
            best_new_coverage = None
            
            for candidate in candidates:
                # Compute what this candidate would cover
                candidate_visibility = self.compute_visibility(candidate)
                
                # Compute marginal coverage (new points that would be covered)
                potential_coverage = np.maximum(coverage, candidate_visibility)
                new_points_covered = np.sum((potential_coverage > self.coverage_threshold) & 
                                          (coverage <= self.coverage_threshold))
                
                # Score based on new coverage
                score = new_points_covered
                
                # Tie-breaking: prefer candidates that improve overall coverage quality
                if new_points_covered > 0:
                    coverage_quality = np.mean(potential_coverage[potential_coverage > self.coverage_threshold])
                    score += coverage_quality * 0.1
                
                if score > best_score:
                    best_score = score
                    best_candidate = candidate.copy()
                    best_new_coverage = candidate_visibility.copy()
            
            # If no improvement possible, break
            if best_candidate is None or best_score <= 0:
                if args.verbose:
                    print("No more improvement possible with current candidates")
                break
                
            # Add best candidate
            selected_viewpoints.append(best_candidate)
            coverage = np.maximum(coverage, best_new_coverage)
            
            iteration += 1
            
        elapsed_time = time.time() - start_time
        final_coverage = np.mean(coverage > self.coverage_threshold)
        
        if args.verbose:
            print(f"\nGreedy Algorithm Complete:")
            print(f"  Total viewpoints: {len(selected_viewpoints)}")
            print(f"  Final coverage: {final_coverage:.4f}")
            print(f"  Total time: {elapsed_time:.2f}s")
        
        return np.array(selected_viewpoints), coverage, total_coverage_history

    def compute_coverage_monte_carlo(self, particles, num_samples=10000, threshold=0.25):
        """Monte Carlo coverage estimation for comparison"""
        pts = np.random.rand(num_samples, 2) * self.grid_size
        diff = pts[:, None, :] - particles[None, :, :]
        d = np.linalg.norm(self.wrap_distance(diff), axis=-1)
        visible = (d <= self.fov_radius).astype(float)
        return np.sum(np.sum(visible, axis=1) > threshold) / num_samples

# === Helper: plot frame ===
def plot_frame(ax, particles, coverage_map, title=None):
    ax.clear()
    
    # Reshape coverage for display
    coverage_display = coverage_map.reshape(grid_size, grid_size)
    
    # Plot coverage map
    im = ax.imshow(coverage_display, cmap='viridis', origin='lower',
                   vmin=0, vmax=1, extent=[0, grid_size, 0, grid_size])
    
    # Plot FOV circles for each viewpoint
    for p in particles:
        for dx in [-grid_size, 0, grid_size]:
            for dy in [-grid_size, 0, grid_size]:
                center = (p[0] + dx, p[1] + dy)
                circle = plt.Circle(center, fov_radius, edgecolor='white', 
                                  facecolor='none', alpha=0.3, linewidth=1)
                ax.add_patch(circle)
    
    # Plot viewpoints
    ax.scatter(particles[:, 0], particles[:, 1], s=50, c='red', 
              edgecolors='white', linewidths=1.5, marker='o')
    
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if title:
        ax.set_title(title, fontsize=12)
    return im

# === Setup ===
os.makedirs(args.save_dir, exist_ok=True)

# Run greedy algorithm
greedy_field = GreedyCoverage(grid_size, fov_radius)
start_time = time.time()

if args.strategy == "greedy":
    viewpoints, final_coverage, coverage_history = greedy_field.greedy_set_cover(
        max_viewpoints=max_viewpoints, target_coverage=0.99)
else:
    # Random baseline for comparison
    viewpoints = np.random.rand(max_viewpoints, 2) * grid_size
    final_coverage = greedy_field.compute_visibility(viewpoints[0])
    for vp in viewpoints[1:]:
        vp_coverage = greedy_field.compute_visibility(vp)
        final_coverage = np.maximum(final_coverage, vp_coverage)
    coverage_history = []

elapsed_time = time.time() - start_time

# Compute final metrics
final_coverage_ratio = np.mean(final_coverage > args.coverage_threshold)
monte_carlo_coverage = greedy_field.compute_coverage_monte_carlo(viewpoints)

print(f"\n=== Final Results ===")
print(f"Strategy: {args.strategy}")
print(f"Number of viewpoints: {len(viewpoints)}")
print(f"Coverage ratio (threshold={args.coverage_threshold}): {final_coverage_ratio:.4f}")
print(f"Monte Carlo coverage: {monte_carlo_coverage:.4f}")
print(f"Total computation time: {elapsed_time:.2f}s")

# === Visualization ===
if args.animate or True:  # Always show final result
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot final coverage
    plot_frame(axes[0], viewpoints, final_coverage, 
               f"{args.strategy.title()} Coverage (Final)")
    
    # Plot coverage history
    if coverage_history:
        axes[1].plot(coverage_history, 'b-', linewidth=2, marker='o')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Coverage Ratio')
        axes[1].set_title('Coverage Progress')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)
    else:
        axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(args.save_dir, f"{args.strategy}_coverage_result.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {save_path}")
    
    # plt.show()

# === Save Results ===
results_data = {
    'strategy': args.strategy,
    'viewpoints': viewpoints,
    'final_coverage': final_coverage,
    'coverage_history': np.array(coverage_history) if coverage_history else np.array([]),
    'final_coverage_ratio': final_coverage_ratio,
    'monte_carlo_coverage': monte_carlo_coverage,
    'num_viewpoints': len(viewpoints),
    'computation_time': elapsed_time,
    'parameters': {
        'grid_size': grid_size,
        'fov_radius': fov_radius,
        'coverage_threshold': args.coverage_threshold,
        'candidate_density': args.candidate_density,
        'k_attr': args.k_attr,
        'k_rep': args.k_rep
    }
}

npz_path = os.path.join(args.save_dir, f"{args.strategy}_coverage_results.npz")
np.savez_compressed(npz_path, **results_data)
print(f"Saved results: {npz_path}")
print("=== Greedy Coverage Algorithm Complete ===")