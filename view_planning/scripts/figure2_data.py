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
parser.add_argument("--strategy", type=str, default="nbv", choices=["nbv", "random"])
parser.add_argument("--potential_type", type=str, default="log")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save_dir", type=str, default="./results2")
parser.add_argument("--animate", action="store_true")
parser.add_argument("--smart_nbv_insertion", action="store_true")
parser.add_argument("--k_attr", type=float, default=0.4)
parser.add_argument("--k_rep", type=float, default=1.0)
parser.add_argument("--verbose", action="store_true")
# New optimization arguments
parser.add_argument("--fast_mode", action="store_true", help="Enable speed optimizations")
parser.add_argument("--monte_carlo_samples", type=int, default=10000, help="Number of Monte Carlo samples")
parser.add_argument("--force_grid_subsample", type=int, default=None, help="Subsample grid points for force computation")
args = parser.parse_args()

# === Constants ===
np.random.seed(args.seed)
grid_size = 100
fov_radius = 20
epsilon = 1e-6
frames_per_stage = 10
max_viewpoints = 100

x = np.arange(grid_size)
y = np.arange(grid_size)
X, Y = np.meshgrid(x, y)
field_points = np.stack([X.ravel(), Y.ravel()], axis=-1)

if args.verbose:
    print(f"Numba JIT available: {NUMBA_AVAILABLE}")
    if args.fast_mode:
        print("Fast mode enabled - using speed optimizations")

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
def fast_monte_carlo_coverage(particles, grid_size, fov_radius, num_samples, threshold=0.25):
    """Fast Monte Carlo coverage estimation"""
    covered_count = 0
    
    for i in range(num_samples):
        # Generate random point
        x = np.random.random() * grid_size
        y = np.random.random() * grid_size
        
        # Count how many particles can see this point
        visible_count = 0
        for j in range(particles.shape[0]):
            dx = x - particles[j, 0]
            dy = y - particles[j, 1]
            
            # Apply wrapping
            if abs(dx) > grid_size / 2:
                dx = -np.sign(dx) * (grid_size - abs(dx))
            if abs(dy) > grid_size / 2:
                dy = -np.sign(dy) * (grid_size - abs(dy))
                
            distance = np.sqrt(dx*dx + dy*dy)
            if distance <= fov_radius:
                visible_count += 1
        
        if visible_count > threshold:
            covered_count += 1
    
    return covered_count / num_samples

# === Optimized Field Class ===
class Field:
    def __init__(self, grid_size, fov_radius, use_wrapping=True):
        self.grid_size = grid_size
        self.fov_radius = fov_radius
        self.use_wrapping = use_wrapping
        self.visibility = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.potential = np.zeros((grid_size, grid_size))
        self.field_points = field_points
        
        # Optimization caches
        self._last_particles_hash = None
        self._cached_visibility = None
        self._cached_potential = None
        
        # Subsampling for force computation
        if args.fast_mode and args.force_grid_subsample:
            # Create subsampled grid for force computation
            n_subsample = min(args.force_grid_subsample, len(field_points))
            self.force_indices = np.random.choice(len(field_points), n_subsample, replace=False)
            self.force_field_points = field_points[self.force_indices]
        else:
            self.force_indices = None
            self.force_field_points = field_points

    def _particles_hash(self, particles):
        """Simple hash for particle positions to detect changes"""
        return hash(particles.tobytes())

    def wrap_distance(self, diff):
        if not self.use_wrapping:
            return diff
        
        if args.fast_mode and NUMBA_AVAILABLE and diff.ndim == 3:
            return fast_wrap_distance(diff, self.grid_size)
        else:
            return np.where(np.abs(diff) > self.grid_size / 2,
                            -np.sign(diff) * (self.grid_size - np.abs(diff)), diff)

    def update_visibility(self, particles, beta=20.0):
        # Cache check for fast mode
        if args.fast_mode:
            particles_hash = self._particles_hash(particles)
            if particles_hash == self._last_particles_hash and self._cached_visibility is not None:
                self.visibility = self._cached_visibility
                return
        
        if args.fast_mode and NUMBA_AVAILABLE:
            visibility_flat = fast_visibility_computation(
                self.field_points, particles, self.grid_size, self.fov_radius, beta
            )
            self.visibility = visibility_flat.reshape(self.grid_size, self.grid_size)
        else:
            # Original implementation
            self.visibility.fill(0)
            diff = self.field_points[:, None, :] - particles[None, :, :]
            wrapped_diff = self.wrap_distance(diff)
            distances = np.linalg.norm(wrapped_diff, axis=-1) + epsilon
            s = 1.0 / (1.0 + np.exp(beta * (distances / self.fov_radius - 1.0)))
            self.visibility.ravel()[:] = np.clip(np.sum(s, axis=1), 0.0, 1.0)
        
        # Update cache
        if args.fast_mode:
            self._last_particles_hash = self._particles_hash(particles)
            self._cached_visibility = self.visibility.copy()

    def compute_potential(self, particles, alpha=1.0):
        # Use cached potential if particles haven't moved much
        if args.fast_mode and self._cached_potential is not None:
            particles_hash = self._particles_hash(particles)
            if particles_hash == self._last_particles_hash:
                self.potential = self._cached_potential
                return
        
        diff = self.field_points[:, None, :] - particles[None, :, :]
        wrapped_diff = self.wrap_distance(diff)
        distances = np.linalg.norm(wrapped_diff, axis=-1) + epsilon
        log_sum = np.log(distances).sum(axis=1)
        covered_factor = 1.0 - self.visibility.ravel()
        pot_values = alpha * covered_factor * log_sum
        self.potential = pot_values.reshape(self.grid_size, self.grid_size)
        
        # Update cache
        if args.fast_mode:
            self._cached_potential = self.potential.copy()

    def compute_force(self, particles, k_attr=0.4, k_rep=1.0, alpha=1.0):
        # Use subsampled field points for force computation if enabled
        field_pts = self.force_field_points
        
        if args.fast_mode and self.force_indices is not None:
            # Use subsampled visibility
            visibility_subset = self.visibility.ravel()[self.force_indices]
        else:
            field_pts = self.field_points
            visibility_subset = self.visibility.ravel()
        
        diff = field_pts[:, None, :] - particles[None, :, :]
        wrapped_diff = self.wrap_distance(diff)
        distances = np.linalg.norm(wrapped_diff, axis=-1) + epsilon
        directions = wrapped_diff / distances[..., None]
        covered_factor = (1.0 - visibility_subset)[:, None]
        coverage_need = alpha * covered_factor / distances
        coverage_need[distances < epsilon] = 0.0
        attractive_forces = coverage_need[..., None] * directions
        
        if args.fast_mode and self.force_indices is not None:
            # For subsampled version, we need to scale the forces appropriately
            scale_factor = len(self.field_points) / len(field_pts)
            attr = attractive_forces.sum(axis=0) * scale_factor
        else:
            attr = attractive_forces.reshape(
                self.grid_size, self.grid_size, particles.shape[0], 2).sum(axis=(0, 1))

        # Repulsive forces (unchanged)
        pairwise_diff = particles[:, None, :] - particles[None, :, :]
        wrapped_diff = self.wrap_distance(pairwise_diff)
        distances = np.linalg.norm(wrapped_diff, axis=-1) + epsilon
        sigma = 10
        amplitude = 100
        forces = -(
            amplitude * wrapped_diff *
            (-distances[..., None] / sigma**2) *
            np.exp(-distances**2 / (2 * sigma**2))[..., None]
        )
        rep = forces.sum(axis=1)

        return k_attr * attr + k_rep * rep

    def compute_coverage_monte_carlo(self, particles, num_samples=None, threshold=0.25):
        if num_samples is None:
            num_samples = args.monte_carlo_samples
        
        # Reduce samples in fast mode for early iterations
        if args.fast_mode:
            if len(particles) < 5:
                num_samples = max(500, num_samples // 10)  # Much fewer samples early on
            elif len(particles) < 10:
                num_samples = max(1000, num_samples // 5)
            else:
                num_samples = max(2000, num_samples // 2)
        
        if args.fast_mode and NUMBA_AVAILABLE:
            return fast_monte_carlo_coverage(particles, self.grid_size, self.fov_radius, num_samples, threshold)
        else:
            # Original implementation
            pts = np.random.rand(num_samples, 2) * self.grid_size
            diff = pts[:, None, :] - particles[None, :, :]
            d = np.linalg.norm(self.wrap_distance(diff), axis=-1)
            visible = (d <= self.fov_radius).astype(float)
            return np.sum(np.sum(visible, axis=1) > threshold) / num_samples

# === Helper: plot frame (unchanged) ===
def plot_frame(ax, particles, potential, nbv_point=None, title=None):
    ax.clear()
    im = ax.imshow(potential, cmap='viridis', origin='lower',
                   vmin=potential.min(), vmax=potential.max(), extent=[0, grid_size, 0, grid_size])
    for p in particles:
        for dx in [-grid_size, 0, grid_size]:
            for dy in [-grid_size, 0, grid_size]:
                center = (p[0] + dx, p[1] + dy)
                circle = plt.Circle(center, fov_radius, edgecolor='white', facecolor='none', alpha=0.2, linewidth=1)
                ax.add_patch(circle)
    ax.scatter(particles[:, 0], particles[:, 1], s=30, c='deepskyblue', edgecolors='white', linewidths=0.6)
    if nbv_point is not None and args.strategy == "nbv":
        ax.scatter(nbv_point[0], nbv_point[1], s=60, c='red', edgecolors='white', linewidths=1.5)
        ax.text(nbv_point[0]+2, nbv_point[1]+2, 'NBV', color='red', fontsize=8, weight='bold')
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if title:
        ax.set_title(title, fontsize=10)
    return im

# === Setup ===
os.makedirs(args.save_dir, exist_ok=True)
field = Field(grid_size, fov_radius)
# from fast_field import Field
field = Field(grid_size, fov_radius)
particles = np.random.rand(1, 2) * grid_size
m = np.zeros_like(particles)
v = np.zeros_like(particles)
t = 0
coverage_data = []
initial_particles = particles.copy()

frames_particles = []
frames_potentials = []
frames_nbvs = []

recent_movements = []
window_size = 5
T_motion = 1.5
iter_count = 0

# === Main Simulation ===
start_time = time.time()

while True:
    field.update_visibility(particles)
    field.compute_potential(particles)
    coverage = field.compute_coverage_monte_carlo(particles)
    coverage_data.append(coverage)

    if args.strategy == "nbv":
        max_idx = np.argmax(field.potential)
        nbv = field.field_points[max_idx]
    else:
        nbv = np.random.rand(2) * grid_size

    # Store frames (reduced frequency in fast mode)
    store_frame = True
    if args.fast_mode and not args.animate:
        # Store fewer frames when not animating in fast mode
        store_frame = (iter_count % 5 == 0)
    
    if store_frame:
        frames_particles.append(particles.copy())
        frames_potentials.append(field.potential.copy())
        frames_nbvs.append(nbv.copy())

    forces = field.compute_force(particles, k_attr=args.k_attr, k_rep=args.k_rep)
    t += 1
    m = 0.9 * m + 0.1 * forces
    v = 0.999 * v + 0.001 * (forces ** 2)
    m_hat = m / (1 - 0.9 ** t)
    v_hat = v / (1 - 0.999 ** t)
    update = 5 * m_hat / (np.sqrt(v_hat) + epsilon)
    particles += update
    particles %= grid_size

    movement = np.linalg.norm(update, axis=1).mean()
    recent_movements.append(movement)
    if len(recent_movements) > window_size:
        recent_movements.pop(0)

    insert_nbv = False
    if args.verbose:
        print(f"[Step {iter_count:03d}] Avg Motion = {movement:.4f} | Coverage = {coverage:.4f} | Viewpoints = {len(particles)}", end='')

    if args.smart_nbv_insertion:
        if len(particles) < 3:
            insert_nbv = (iter_count % frames_per_stage == 0)
            if args.verbose:
                print(f" | Insert (warm-up) = {'✔️' if insert_nbv else '❌'}")
        else:
            settled = len(recent_movements) == window_size and all(m < T_motion for m in recent_movements)
            insert_nbv = settled
            if args.verbose:
                print(f" | Settled = {'✔️' if settled else '❌'}")
    else:
        insert_nbv = (iter_count % frames_per_stage == 0)
        if args.verbose:
            print(f" | Insert (fixed rate) = {'✔️' if insert_nbv else '❌'}")

    iter_count += 1

    if insert_nbv:
        particles = np.vstack([particles, nbv[None, :]])
        m = np.zeros_like(particles)
        v = np.zeros_like(particles)
        t = 0
        recent_movements.clear()

    if coverage >= 0.99 or len(particles) >= max_viewpoints:
        break

elapsed_time = time.time() - start_time

# === Save Snapshots (unchanged) ===
if args.animate:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    field.update_visibility(initial_particles)
    field.compute_potential(initial_particles)
    im0 = plot_frame(axes[0], initial_particles, field.potential, title="Initial")

    field.update_visibility(particles)
    field.compute_potential(particles)
    im1 = plot_frame(axes[1], particles, field.potential, title="Final")

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label("Potential")

    snapshot_path = os.path.join(args.save_dir, f"{args.strategy}_{args.potential_type}_seed{args.seed}_snapshots.png")
    plt.savefig(snapshot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Saved] Initial/Final snapshot at: {snapshot_path}")

# === Save Animation (unchanged) ===
if args.animate:
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('black')

    def animate(i):
        return [plot_frame(ax, frames_particles[i], frames_potentials[i], frames_nbvs[i])]

    ani = animation.FuncAnimation(fig, animate, frames=len(frames_particles), interval=100, blit=False)
    mp4_path = os.path.join(args.save_dir, f"{args.strategy}_{args.potential_type}_seed{args.seed}_anim.mp4")
    ani.save(mp4_path, writer='ffmpeg', fps=10, dpi=300)
    plt.close()
    print(f"[Saved] Animation MP4 at: {mp4_path}")

# === Compute Redundancy and Affinity (unchanged) ===
redundancy_data = []
affinity_data = []

def compute_redundancy(particles, num_samples=10000):
    random_points = np.random.rand(num_samples, 2) * grid_size
    diff = random_points[:, None, :] - particles[None, :, :]
    wrapped_diff = field.wrap_distance(diff)
    distances = np.linalg.norm(wrapped_diff, axis=-1)
    visibility = (distances <= fov_radius).astype(float)
    counts = np.sum(visibility, axis=1)
    redundant = np.sum(counts > 1)
    return redundant / num_samples

def compute_overlap_affinity(particles, num_samples=10000):
    random_points = np.random.rand(num_samples, 2) * grid_size
    diff = random_points[:, None, :] - particles[None, :, :]
    wrapped_diff = field.wrap_distance(diff)
    distances = np.linalg.norm(wrapped_diff, axis=-1)
    visibility = (distances <= fov_radius).astype(float)
    counts = np.sum(visibility, axis=1)
    visible = counts > 0
    return np.mean(counts[visible]) if np.any(visible) else 0.0

# Updated loop
for i in range(len(frames_particles)):
    pts = frames_particles[i]
    redundancy = compute_redundancy(pts)
    affinity = compute_overlap_affinity(pts)
    redundancy_data.append(redundancy)
    affinity_data.append(affinity)


# === Save Results (unchanged) ===
npz_path = os.path.join(args.save_dir, f"{args.strategy}_{args.potential_type}_kattr{args.k_attr:.2f}_krep{args.k_rep:.2f}_seed{args.seed}_metrics.npz")
np.savez_compressed(npz_path,
    coverage=np.array(coverage_data),
    redundancy=np.array(redundancy_data),
    affinity=np.array(affinity_data),
    num_viewpoints=len(particles),
    num_iterations=iter_count,
    simulation_time_sec=elapsed_time
)

print(f"[Saved] Metrics NPZ at: {npz_path}")
print("[Done] Simulation complete.")

if args.verbose:
    print(f"\nPerformance Summary:")
    print(f"  Total iterations: {iter_count}")
    print(f"  Total time: {elapsed_time:.2f}s")
    print(f"  Time per iteration: {elapsed_time/iter_count:.3f}s")
    print(f"  Fast mode: {args.fast_mode}")
    print(f"  Numba available: {NUMBA_AVAILABLE}")